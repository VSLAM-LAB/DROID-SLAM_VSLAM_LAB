from pathlib import Path
from tqdm import tqdm
import pkg_resources
import pandas as pd
import numpy as np
import argparse
import torch
import yaml
import time
import csv
import cv2
import sys
import os

sys.path.append('droid_slam')
from droid_slam.droid import Droid

timestamps = []
DROID_STEREO_BASELINE_M = 0.1

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def load_calibration(calibration_yaml: Path, cam_l_name: str, cam_r_name: str, target_pixels: int = 384*512):
    with open(calibration_yaml, 'r') as file:
        data = yaml.safe_load(file)
    cameras = data.get('cameras', [])
    for cam_ in cameras:
        if cam_['cam_name'] == cam_l_name:
            cam_l = cam_
        if cam_['cam_name'] == cam_r_name:
            cam_r = cam_
    
    K, D, T_BS, R_BS, t_BS = {}, {} ,{}, {}, {}
    for cam, side in zip([cam_l, cam_r],['l', 'r']):
        print(f"\nCamera Name: {cam['cam_name']}")
        print(f"Camera Type: {cam['cam_type']}")
        print(f"Camera Model: {cam['cam_model']}")
        print(f"Focal Length: {cam['focal_length']}")
        print(f"Principal Point: {cam['principal_point']}")
        has_dist = ('distortion_type' in cam) and ('distortion_coefficients' in cam)
        if has_dist:
            print(f"Distortion Type Dimension: {cam['distortion_type']}")
            print(f"Distortion Coefficients: {cam['distortion_coefficients']}")
        print(f"Image Dimension: {cam['image_dimension']}")
        print(f"Fps: {cam['fps']}")

        K[side] = np.array([[cam['focal_length'][0], 0,  cam['principal_point'][0]],
                    [0,  cam['focal_length'][1], cam['principal_point'][1]],
                    [0,  0,   1]], dtype=np.float32)
        
        if ('distortion_type' in cam) and ('distortion_coefficients' in cam):
            D[side] = np.array(cam['distortion_coefficients'], dtype=np.float32)
        else:
            D[side] = 0.0

        T_BS[side] = np.array(cam['T_BS']).reshape(4, 4)  
        R_BS[side], t_BS[side] = T_BS[side][:3, :3], T_BS[side][:3, 3]

    R_01_B = R_BS['r'].T @ R_BS['l']
    t_01_B = R_BS['r'].T @ (t_BS['l'] - t_BS['r'])

    w0 = cam_l['image_dimension'][0]
    h0 = cam_l['image_dimension'][1]
    h = (int(h0 * np.sqrt(target_pixels / (h0 * w0))) // 32) * 32
    w = (int(w0 * np.sqrt(target_pixels / (h0 * w0))) // 32) * 32

    R_l, R_r, P_l, P_r, Q, _, _ = cv2.stereoRectify(
        K['l'], D['l'], K['r'], D['r'], (w0, h0), R_01_B.astype(np.float64), t_01_B.astype(np.float64),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=(w, h)
    )
    map_l = cv2.initUndistortRectifyMap(K['l'], D['l'], R_l, P_l[:3,:3], (w, h), cv2.CV_32F)
    map_r = cv2.initUndistortRectifyMap(K['r'], D['r'], R_r, P_r[:3,:3], (w, h), cv2.CV_32F)

    baseline = float(np.linalg.norm(t_01_B))
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError(f"Invalid stereo baseline computed from calibration: {baseline}")

    return P_l[0:3,0:3], map_l, map_r, baseline

def image_stream(sequence_path: Path, rgb_csv: Path, calibration_yaml: Path, 
                 cam_l_name = "rgb_0" , cam_r_name = "rgb_1", target_pixels: int = 384*512,
                 calibration=None):
    """ image generator """ 
    global timestamps 
    if calibration is None:
        calibration = load_calibration(calibration_yaml=calibration_yaml,
                                       cam_l_name=cam_l_name, cam_r_name=cam_r_name,
                                       target_pixels=target_pixels)
    K, map_l, map_r, _ = calibration

    # Load rgb images
    df = pd.read_csv(rgb_csv)       
    images_left = df[f'path_{cam_l_name}'].to_list()
    images_right = df[f'path_{cam_r_name}'].to_list()
    timestamps = (df[f'ts_{cam_l_name} (ns)'] / 1e9).to_list()

    intrinsics_vec = [K[0,0], K[1,1], K[0,2], K[1,2]]
    intrinsics = torch.as_tensor(intrinsics_vec).cuda()
    
    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        imgL = os.path.join(sequence_path, imgL)
        imgR = os.path.join(sequence_path, imgR)

        images = [cv2.remap(cv2.imread(imgL), map_l[0], map_l[1], interpolation=cv2.INTER_LINEAR)]
        images += [cv2.remap(cv2.imread(imgR), map_r[0], map_r[1], interpolation=cv2.INTER_LINEAR)]

        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)

        yield t, images, intrinsics.clone()
    
def main():    
    print("\nRunning vslamlab_droidslam_stereo.py ...\n")  

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--calibration_yaml", type=Path, required=True)
    parser.add_argument("--rgb_csv", type=Path, required=True)
    parser.add_argument("--exp_folder", type=Path, required=True)
    parser.add_argument("--exp_it", type=str, default="0")
    parser.add_argument("--settings_yaml", type=Path, default=None)
    parser.add_argument("--verbose", type=str, help="verbose")
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--weights", type=Path, default=None)

    args, _ = parser.parse_known_args()

    verbose = bool(int(args.verbose))
    args.disable_vis = not bool(int(args.verbose))
    args.upsample = bool(int(args.upsample))

    settings_path = args.settings_yaml
    if not os.path.exists(settings_path):
        settings_path = pkg_resources.resource_filename(
            'droid_slam.configs', 'vslamlab_droidslam-dev_settings.yaml'
        )

    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)
    S = settings.get('settings', {})

    # Attach required settings to args
    args.t0 = int(S.get('t0', 0))
    args.stride = int(S.get('stride', 3))
    args.buffer = int(S.get('buffer', 512))
    args.beta = float(S.get('beta', 0.3))
    args.filter_thresh = float(S.get('filter_thresh', 2.4))
    args.warmup = int(S.get('warmup', 8))
    args.keyframe_thresh = float(S.get('keyframe_thresh', 4.0))
    args.frontend_thresh = float(S.get('frontend_thresh', 16.0))
    args.frontend_window = int(S.get('frontend_window', 25))
    args.frontend_radius = int(S.get('frontend_radius', 2))
    args.frontend_nms = int(S.get('frontend_nms', 1))
    args.backend_thresh = float(S.get('backend_thresh', 22.0))
    args.backend_radius = int(S.get('backend_radius', 2))
    args.backend_nms = int(S.get('backend_nms', 3))

    args.stereo = True
    args.depth = False
    cam_names = S.get('cam_stereo',  ["rgb_0", "rgb_1"])
    calibration = load_calibration(calibration_yaml=args.calibration_yaml,
                                   cam_l_name=cam_names[0], cam_r_name=cam_names[1])
    baseline = calibration[3]
    translation_scale = baseline / DROID_STEREO_BASELINE_M
    print(f"Stereo baseline: {baseline:.6f} m; translation scale: {translation_scale:.6f}")
    
    torch.multiprocessing.set_start_method('spawn')

    droid = None
    for (t, image, intrinsics) in tqdm(image_stream(args.sequence_path, args.rgb_csv, args.calibration_yaml,
                                                    cam_l_name=cam_names[0], cam_r_name=cam_names[1],
                                                    calibration=calibration)):
        if t < args.t0:
            continue

        if verbose:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]         
            droid = Droid(args)
            time.sleep(5)

        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.sequence_path, args.rgb_csv, args.calibration_yaml,
                                            cam_l_name=cam_names[0], cam_r_name=cam_names[1],
                                            calibration=calibration))
    traj_est[:, :3] *= translation_scale
    
    keyframe_csv = args.exp_folder / f"{args.exp_it.zfill(5)}_KeyFrameTrajectory.csv"
    with open(keyframe_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
        for i in range(len(timestamps)):
            ts = int(timestamps[i] * 1e9)
            tx, ty, tz, qx, qy, qz, qw = traj_est[i][:7]
            writer.writerow([ts, tx, ty, tz, qx, qy, qz, qw])

    time.sleep(10)
    import signal
    os.killpg(os.getpgrp(), signal.SIGTERM)

if __name__ == '__main__':
    main()
