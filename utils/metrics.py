import os
from typing import List, Union
from einops import rearrange
from skimage.metrics import structural_similarity
import numpy as np
import torch


def parse_checkpoints(checkpoints: Union[str, int], max_steps: int) -> List[int]:
    if checkpoints == "none":
        checkpoints = [max_steps]
    elif "every" in checkpoints:
        # e.g. every_10000
        _, interval = checkpoints.split("_")
        interval = int(interval)
        checkpoints = list(range(interval, max_steps, interval))
        checkpoints.append(max_steps)
    elif isinstance(checkpoints, int):
        # e.g. 40000
        if checkpoints >= max_steps:
            checkpoints = [max_steps]
        else:
            checkpoints = [checkpoints, max_steps]
    else:
        # e.g. [40000,50000,60000]
        checkpoints = [int(s) for s in checkpoints.split(",") if int(s) < max_steps]
        checkpoints.append(max_steps)
    return checkpoints


def get_type_max(data):
    dtype = data.dtype.name
    if dtype == "uint8":
        max = 255
    elif dtype == "uint12":
        max = 4098
    elif dtype == "uint16":
        max = 65535
    elif dtype == "float32":
        max = 65535
    elif dtype == "float64":
        max = 65535
    else:
        raise NotImplementedError
    return max


def calc_psnr(gt: np.ndarray, predicted: np.ndarray):
    data_range = get_type_max(gt)
    mse = np.mean(np.power(predicted / data_range - gt / data_range, 2))
    psnr = -10 * np.log10(mse)
    return psnr


def calc_ssim(gt: np.ndarray, predicted: np.ndarray):
    data_range = get_type_max(gt)
    ssim = structural_similarity(gt, predicted, data_range=data_range)
    return ssim


def get_folder_size(folder_path: str):
    total_size = 0
    if os.path.isdir(folder_path):
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    else:
        total_size = os.path.getsize(folder_path)
    return total_size
