from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths, library_paths
import os.path as osp
import torch
import os

ROOT = osp.dirname(osp.abspath(__file__))
torch_include_dirs = include_paths()
torch_library_dirs = library_paths()
conda_prefix = os.environ.get("PREFIX", os.environ.get("CONDA_PREFIX", ""))
eigen_path = osp.join(conda_prefix, 'include', 'eigen3')

setup(
    name='vslamlab_droidslam',
    version='0.1',
    description='DROID-SLAM',
    package_data={
        'droid_slam.configs': ['*.yaml'],
    },
    include_package_data=True,
    py_modules=['vslamlab_droidslam_mono', 'vslamlab_droidslam_rgbd', 'vslamlab_droidslam_stereo'],
    packages=find_packages(where='.'),
    package_dir={
        'droid_slam': 'droid_slam',
    },
    entry_points={
        'console_scripts': [
            'vslamlab_droidslam_mono = vslamlab_droidslam_mono:main',
            'vslamlab_droidslam_rgbd = vslamlab_droidslam_rgbd:main',
            'vslamlab_droidslam_stereo = vslamlab_droidslam_stereo:main',
        ]
    },
    ext_modules=[
        CUDAExtension(
            name='droid_backends',
            include_dirs=torch_include_dirs + [
                eigen_path,
                osp.join(ROOT, 'src')
            ],
            library_dirs=torch_library_dirs,
            sources=[
                'src/droid.cpp',
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-D_GLIBCXX_USE_CXX11_ABI=1'],
                'nvcc': [
                    '-O3',
                    '-D_GLIBCXX_USE_CXX11_ABI=1',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_90,code=sm_90',
                    '-gencode=arch=compute_90,code=compute_90',
                ]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
)
