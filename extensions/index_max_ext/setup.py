import setuptools
import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(name='index_max',
      ext_modules=[CUDAExtension('index_max', ['index_max.cpp', 'index_max_cuda.cu'])],
      cmdclass={'build_ext': BuildExtension})