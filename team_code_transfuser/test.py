import copy
import glob
import importlib
import importlib.abc
import os
import re
import shlex
import setuptools
import subprocess
import sys
import sysconfig
import warnings
import collections

import torch
from typing import List, Optional, Union, Tuple

IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')
IS_LINUX = sys.platform.startswith('linux')
LIB_EXT = '.pyd' if IS_WINDOWS else '.so'
EXEC_EXT = '.exe' if IS_WINDOWS else ''
CLIB_PREFIX = '' if IS_WINDOWS else 'lib'
CLIB_EXT = '.dll' if IS_WINDOWS else '.so'
SHARED_FLAG = '/DLL' if IS_WINDOWS else '-shared'

_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
TORCH_LIB_PATH = os.path.join(_TORCH_PATH, 'lib')


BUILD_SPLIT_CUDA = os.getenv('BUILD_SPLIT_CUDA') or (os.path.exists(os.path.join(
    TORCH_LIB_PATH, f'{CLIB_PREFIX}torch_cuda_cu{CLIB_EXT}')) and os.path.exists(os.path.join(TORCH_LIB_PATH, f'{CLIB_PREFIX}torch_cuda_cpp{CLIB_EXT}')))


def _find_cuda_home() -> Optional[str]:
    r'''Finds the CUDA install path.'''
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output([which, 'nvcc'],
                                               stderr=devnull).decode(*SUBPROCESS_DECODE_ARGS).rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and not torch.cuda.is_available():
        print(f"No CUDA runtime is found, using CUDA_HOME='{cuda_home}'",
              file=sys.stderr)
    return cuda_home

CUDA_HOME = _find_cuda_home()

def test():
    if not CUDA_HOME:
        raise RuntimeError(CUDA_NOT_FOUND_MESSAGE)

    nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
    cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
    cuda_version = re.search(r'release (\d+[.]\d+)', cuda_version_str)
    if cuda_version is None:
        print("cuda_version is None")

    cuda_str_version = cuda_version.group(1)
    cuda_ver = packaging.version.parse(cuda_str_version)
    torch_cuda_version = packaging.version.parse(torch.version.cuda)
    if cuda_ver != torch_cuda_version:
        # major/minor attributes are only available in setuptools>=49.6.0
        print(cuda_ver)
        print(torch_cuda_version)

test()