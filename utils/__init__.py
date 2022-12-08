import csv
from dataclasses import dataclass
from datetime import datetime
import math
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
import random
from typing import List, Union
import numpy as np
from scipy import ndimage
import torch


@dataclass
class SideInfos3D:
    dtype: str = ""
    depth: int = 0
    height: int = 0
    width: int = 0
    original_min: int = 0
    original_max: int = 0
    normalized_min: int = 0
    normalized_max: int = 0


@dataclass
class SideInfos4D:
    dtype: str = ""
    time: int = 0
    depth: int = 0
    height: int = 0
    width: int = 0
    original_min: int = 0
    original_max: int = 0
    normalized_min: int = 0
    normalized_max: int = 0


def denoise(
    data: np.ndarray,
    denoise_level: int,
    denoise_close: Union[bool, List[int]],
) -> np.ndarray:
    denoised_data = np.copy(data)
    if denoise_close == False:
        # using 'denoise_level' as a hard threshold,
        # the pixel with instensity below this threshold will be set to zero
        denoised_data[data <= denoise_level] = 0
    else:
        # using 'denoise_level' as a soft threshold,
        # only the pixel with itself and neighbors instensities below this threshold will be set to zero
        denoised_data[
            ndimage.binary_opening(
                data <= denoise_level,
                structure=np.ones(tuple(list(denoise_close) + [1])),
                iterations=1,
            )
        ] = 0
    return denoised_data


def normalize(
    data: np.ndarray, sideinfos: Union[SideInfos3D, SideInfos4D]
) -> np.ndarray:
    """
    use minmax normalization to scale and offset the data range to [normalized_min,normalized_max]
    """
    normalized_min, normalized_max = sideinfos.normalized_min, sideinfos.normalized_max
    dtype = data.dtype.name
    data = data.astype(np.float32)
    original_min = float(data.min())
    original_max = float(data.max())
    data = (data - original_min) / (original_max - original_min)
    data *= normalized_max - normalized_min
    data += normalized_min
    sideinfos.dtype = dtype
    sideinfos.original_min = original_min
    sideinfos.original_max = original_max
    return data


def inv_normalize(
    data: np.ndarray, sideinfos: Union[SideInfos3D, SideInfos4D]
) -> np.ndarray:
    dtype = sideinfos.dtype
    if dtype == "uint8":
        dtype = np.uint8
    elif dtype == "uint12":
        dtype = np.uint12
    elif dtype == "uint16":
        dtype = np.uint16
    elif dtype == "float32":
        dtype = np.float32
    elif dtype == "float64":
        dtype = np.float64
    else:
        raise NotImplementedError
    data -= sideinfos.normalized_min
    data /= sideinfos.normalized_max - sideinfos.normalized_min
    data = np.clip(data, 0, 1)
    data = (
        data * (sideinfos.original_max - sideinfos.original_min)
        + sideinfos.original_min
    )
    data = np.array(data, dtype=dtype)
    return data


def generate_weight_map(data: np.ndarray, weight_map_rules: Union[List[str], None]):
    """
    generate weight_map from denoised_data according to a list of wieght_map_rule.
    weight_map will determine the weights of different pixels in the loss function
    in the compression optimization problem.
    """
    weight_map = np.ones_like(data).astype(np.float32)
    if weight_map_rules is not None:
        for weight_map_rule in weight_map_rules:
            if "value" in weight_map_rule:
                # e.g. value_0_2000_0.01
                _, l, h, scale = weight_map_rule.split("_")
                l, h, scale = float(l), float(h), float(scale)
                # l, h = range_limit(data, [l, h])
                weight_map[(data >= l) * (data <= h)] = scale
            else:
                raise NotImplementedError
    return weight_map


def reproduc(seed):
    """make experiments reproducible"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
