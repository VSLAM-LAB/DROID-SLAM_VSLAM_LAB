
import pickle
import os
import os.path as osp

from loguru import logger


# RGBD-Dataset
from .tartan import TartanAir
from .vslamlab import VSLAMLAB, VSLAMLABGTFree

from .stream import ImageStream
from .stream import StereoStream
from .stream import RGBDStream

# streaming datasets for inference
from .tartan import TartanAirStream
from .tartan import TartanAirTestStream

def dataset_factory(dataset_list, scenes_gt=None, scenes_gt_free=None, **kwargs):
    """ create a combined dataset

    scenes_gt / scenes_gt_free: scene-list yamls (relative to datapath) for the
    'vslamlab' / 'vslamlab_gt_free' entries; None keeps each class's default
    ('gt.yaml' / 'gt_free.yaml'). Only forwarded to the VSLAM-LAB readers since
    TartanAir has no such argument. """
    logger.info("=" * 60)
    logger.info(f"[dataset_factory] called with dataset_list={dataset_list}")
    logger.info("=" * 60)

    from torch.utils.data import ConcatDataset

    dataset_map = { 'tartan': (TartanAir, {}),
                    'vslamlab': (VSLAMLAB, {'scenes_yaml': scenes_gt}),
                    'vslamlab_gt_free': (VSLAMLABGTFree, {'scenes_yaml': scenes_gt_free}) }
    db_list = []
    for key in dataset_list:
        cls, extra = dataset_map[key]
        extra = {k: v for k, v in extra.items() if v is not None}
        # cache datasets for faster future loading
        db = cls(**kwargs, **extra)
        db_list.append(db)

    return ConcatDataset(db_list)


def create_datastream(dataset_path, **kwargs):
    """ create data_loader to stream images 1 by 1 """

    from torch.utils.data import DataLoader

    if osp.isfile(osp.join(dataset_path, 'calibration.txt')):
        db = ETH3DStream(dataset_path, **kwargs)

    elif osp.isdir(osp.join(dataset_path, 'image_left')):
        db = TartanAirStream(dataset_path, **kwargs)

    elif osp.isfile(osp.join(dataset_path, 'rgb.txt')):
        db = TUMStream(dataset_path, **kwargs)

    elif osp.isdir(osp.join(dataset_path, 'mav0')):
        db = EurocStream(dataset_path, **kwargs)

    elif osp.isfile(osp.join(dataset_path, 'calib.txt')):
        db = KITTIStream(dataset_path, **kwargs)

    else:
        # db = TartanAirStream(dataset_path, **kwargs)
        db = TartanAirTestStream(dataset_path, **kwargs)

    stream = DataLoader(db, shuffle=False, batch_size=1, num_workers=4)
    return stream


def create_imagestream(dataset_path, **kwargs):
    """ create data_loader to stream images 1 by 1 """
    from torch.utils.data import DataLoader

    db = ImageStream(dataset_path, **kwargs)
    return DataLoader(db, shuffle=False, batch_size=1, num_workers=4)

def create_stereostream(dataset_path, **kwargs):
    """ create data_loader to stream images 1 by 1 """
    from torch.utils.data import DataLoader

    db = StereoStream(dataset_path, **kwargs)
    return DataLoader(db, shuffle=False, batch_size=1, num_workers=4)

def create_rgbdstream(dataset_path, **kwargs):
    """ create data_loader to stream images 1 by 1 """
    from torch.utils.data import DataLoader

    db = RGBDStream(dataset_path, **kwargs)
    return DataLoader(db, shuffle=False, batch_size=1, num_workers=4)

