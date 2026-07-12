
import numpy as np
import yaml
import torch
import glob
import cv2
import os
import os.path as osp
from pathlib import Path

from loguru import logger

from lietorch import SE3
from .base import RGBDDataset
from .stream import RGBDStream

cur_path = osp.dirname(osp.abspath(__file__))
test_split = osp.join(cur_path, 'tartan_test.txt')
test_split = open(test_split).read().split()


class VSLAMLAB(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    # depth png values are stored as depth_meters * DEPTH_FACTOR (see dataset_tartanair_train.yaml)
    depth_factor_cached = {}
    depth_saturation_cached = {}

    # saturated pixels (raw >= 65535) are far/background points whose true depth is
    # unknown but at least the saturation cap; treat them as far (~0 disparity), not near
    DEPTH_FAR = 1e3

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(VSLAMLAB, self).__init__(name='vslamlab', **kwargs)

    @staticmethod
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self):
        from tqdm import tqdm

        scene_info = {}

        import yaml
        scenes_yaml = osp.join(self.root, 'scenes.yaml')
        with open(scenes_yaml, 'r') as f:
            scenes_config = yaml.safe_load(f)

        import pandas as pd
        for key, sequences in scenes_config.items():
            for scene in sequences:
                scene_path = Path(osp.join(self.root, key.upper(), scene))

                rgb_csv = scene_path / 'rgb.csv'
                logger.info(f"Processing scene: {scene_path}")
                df = pd.read_csv(rgb_csv)
                image_list = df[f'path_rgb_0'].to_list()
                depth_list = df[f'path_depth_0'].to_list()
                images = [scene_path / image for image in image_list]
                depths = [scene_path / depth for depth in depth_list]

                gt_csv = scene_path / 'groundtruth.csv'
                gt_df = pd.read_csv(gt_csv)
                poses = gt_df[['tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw']].values

                poses[:,:3] /= VSLAMLAB.DEPTH_SCALE

                calibration_yaml = scene_path / 'calibration.yaml'
                calibration = VSLAMLAB.calib_read(calibration_yaml)
                intrinsics = [calibration[:4]] * len(images)

                # graph of co-visible frames based on flow
                graph = self.build_frame_graph(poses, depths, intrinsics)

                scene = '/'.join(scene.split('/'))
                scene_info[scene] = {'images': images, 'depths': depths,
                    'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

                fx, fy, cx, cy, depth_factor = calibration
                saturation = (65535.0 / depth_factor) / VSLAMLAB.DEPTH_SCALE
                logger.info(f"Scene: {scene_path} | num_rgb={len(images)} num_depth={len(depths)} "
                            f"num_poses={len(poses)} fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} "
                            f"depth_factor={depth_factor} DEPTH_SCALE={VSLAMLAB.DEPTH_SCALE} saturation={saturation:.4f}")
        return scene_info

    @staticmethod
    def calib_read(calibration_yaml: Path):
        cam_name = "rgb_0"
        with open(calibration_yaml, 'r') as file:
            data = yaml.safe_load(file)
        cameras = data.get('cameras', [])
        for cam_ in cameras:
            if cam_['cam_name'] == cam_name:
                cam = cam_
                break

        has_dist = ('distortion_type' in cam) and ('distortion_coefficients' in cam)
        if has_dist:
            dist = np.array(cam['distortion_coefficients'], dtype=np.float32)

        return cam['focal_length'][0], cam['focal_length'][1], cam['principal_point'][0], cam['principal_point'][1], cam['depth_factor']

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def _get_depth_calibration(depth_file):
        """ depth_factor/saturation depend on the scene's calibration.yaml; cache per
        depth folder (e.g. <scene_path>/depth_0) so this works whether or not
        _build_dataset ran this process (it's skipped when scene_info loads from a
        pickle cache, see RGBDDataset.__init__) """
        depth_folder = Path(depth_file).parent
        if depth_folder not in VSLAMLAB.depth_factor_cached:
            calibration_yaml = depth_folder.parent / 'calibration.yaml'
            calibration = VSLAMLAB.calib_read(calibration_yaml)
            depth_factor = calibration[4]
            VSLAMLAB.depth_factor_cached[depth_folder] = depth_factor
            VSLAMLAB.depth_saturation_cached[depth_folder] = (65535.0 / depth_factor) / VSLAMLAB.DEPTH_SCALE
        return VSLAMLAB.depth_factor_cached[depth_folder], VSLAMLAB.depth_saturation_cached[depth_folder]

    @staticmethod
    def depth_read(depth_file):
        depth_factor, saturation = VSLAMLAB._get_depth_calibration(depth_file)

        raw = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth = (raw / depth_factor) / VSLAMLAB.DEPTH_SCALE

        depth[depth==np.inf] = 1.0
        depth[depth >= saturation] = VSLAMLAB.DEPTH_FAR
        # raw == 0 is the standard "no data" sentinel for real depth sensors (occlusion,
        # reflective/out-of-range surfaces); treat as far/low-parallax, same as saturation,
        # rather than near, since we have no information about the true depth
        depth[raw <= 0] = VSLAMLAB.DEPTH_FAR
        return depth

class VSLAMLABStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(VSLAMLABStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/TartanAir'

        scene = osp.join(self.root, self.datapath)
        image_glob = osp.join(scene, 'image_left/*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)


class VSLAMLABTestStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(VSLAMLABTestStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/mono'
        image_glob = osp.join(self.root, self.datapath, '*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(self.root, 'mono_gt', self.datapath + '.txt'), delimiter=' ')
        poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)