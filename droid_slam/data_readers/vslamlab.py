
import hashlib
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

    def __init__(self, mode='training', scenes_yaml='gt.yaml', cache_suffix='', **kwargs):
        self.mode = mode
        self.n_frames = 2
        self.scenes_yaml = scenes_yaml

        # cache is named after the scene-list yaml (eth.yaml -> cache/eth.pickle);
        # the yaml's content hash is stored inside the pickle as a staleness stamp so
        # an edited yaml triggers a rebuild instead of silently loading the old graph
        yaml_path = osp.join(kwargs['datapath'], scenes_yaml)
        with open(yaml_path, 'rb') as f:
            yaml_hash = hashlib.md5(f.read()).hexdigest()
        name = Path(scenes_yaml).stem + cache_suffix

        super(VSLAMLAB, self).__init__(name=name, cache_stamp=yaml_hash, **kwargs)

    @staticmethod
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self):
        from tqdm import tqdm

        scene_info = {}

        import yaml
        scenes_yaml = osp.join(self.root, self.scenes_yaml)
        with open(scenes_yaml, 'r') as f:
            scenes_config = yaml.safe_load(f)

        for key, sequences in scenes_config.items():
            for scene in sequences:
                scene_path = Path(osp.join(self.root, key.upper(), scene))

                rgb_csv = scene_path / 'rgb.csv'
                logger.info(f"Processing scene: {scene_path}")
                import pandas as pd
                df = pd.read_csv(rgb_csv)
                image_list = df[f'path_rgb_0'].to_list()
                depth_list = df[f'path_depth_0'].to_list()
                images = [scene_path / image for image in image_list]
                depths = [scene_path / depth for depth in depth_list]

                poses = self._load_poses(scene_path, len(images))

                calibration_yaml = scene_path / 'calibration.yaml'
                calibration = VSLAMLAB.calib_read(calibration_yaml)
                intrinsics = [calibration[:4]] * len(images)

                graph = self._build_graph(poses, depths, intrinsics)

                # key by dataset + scene so same-named scenes from different
                # datasets can't collide
                scene_key = f"{key.upper()}/{scene}"
                scene_info[scene_key] = {'images': images, 'depths': depths,
                    'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

                fx, fy, cx, cy, depth_factor = calibration
                saturation = (65535.0 / depth_factor) / VSLAMLAB.DEPTH_SCALE
                logger.info(f"Scene: {scene_path} | num_rgb={len(images)} num_depth={len(depths)} "
                            f"num_poses={len(poses)} fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} "
                            f"depth_factor={depth_factor} DEPTH_SCALE={VSLAMLAB.DEPTH_SCALE} saturation={saturation:.4f}")
        return scene_info

    def _load_poses(self, scene_path, num_frames):
        """ gt path: read groundtruth.csv (assumed row-aligned with rgb.csv, see synch_gt) """
        import pandas as pd
        gt_csv = scene_path / 'groundtruth.csv'
        gt_df = pd.read_csv(gt_csv)
        poses = gt_df[['tx (m)', 'ty (m)', 'tz (m)', 'qx', 'qy', 'qz', 'qw']].values
        poses[:, :3] /= VSLAMLAB.DEPTH_SCALE
        return poses

    def _build_graph(self, poses, depths, intrinsics):
        """ gt path: co-visibility graph from gt-pose/depth induced flow """
        return self.build_frame_graph(poses, depths, intrinsics)

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


class VSLAMLABGTFree(VSLAMLAB):
    """ Scenes trained WITHOUT groundtruth (consistency loss only): groundtruth.csv
    is never read — poses are identity placeholders (shape compatibility only; they
    must never feed a supervised loss) and the co-visibility graph is temporal
    rather than flow-based. """

    # temporal-graph neighbors: i +/- TEMPORAL_STRIDE * {1..TEMPORAL_K}
    TEMPORAL_STRIDE = 2
    TEMPORAL_K = 4

    def __init__(self, scenes_yaml='gt_free.yaml', **kwargs):
        # separate cache from the labeled reader of the same yaml (eth.yaml ->
        # eth_gt_free.pickle): identity poses + temporal graph, not gt + flow graph
        super(VSLAMLABGTFree, self).__init__(scenes_yaml=scenes_yaml, cache_suffix='_gt_free', **kwargs)

    def _load_poses(self, scene_path, num_frames):
        # identity poses (tx ty tz qx qy qz qw)
        poses = np.zeros((num_frames, 7), dtype=np.float64)
        poses[:, 6] = 1.0
        return poses

    def _build_graph(self, poses, depths, intrinsics):
        """ temporal graph with synthetic distances 16*m for the m-th neighbor, chosen
        to sit inside the (fmin, fmax) sampling window used in base.py __getitem__;
        interior frames get 2*TEMPORAL_K neighbors, enough to pass the
        len(graph[i][0]) > n_frames anchor check for n_frames=7 """
        N = len(depths)
        graph = {}
        for i in range(N):
            js, ds = [], []
            for m in range(1, self.TEMPORAL_K + 1):
                for j in (i - self.TEMPORAL_STRIDE * m, i + self.TEMPORAL_STRIDE * m):
                    if 0 <= j < N:
                        js.append(j)
                        ds.append(16.0 * m)
            graph[i] = (np.array(js), np.array(ds, dtype=np.float32))
        return graph


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