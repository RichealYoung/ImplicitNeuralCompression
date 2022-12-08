from einops import rearrange
import numpy as np
import tifffile
import nibabel
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

if __name__ == "__main__":
    data_path = "dataset/brain-fmri.nii.gz"
    data_name = opb(data_path).replace(".nii.gz", "")
    original_data_nii = nibabel.load(data_path)
    dtype = original_data_nii.get_data_dtype()
    original_data = original_data_nii.get_fdata().astype(np.float32)
    if dtype == np.int8:
        original_data = original_data + 32768
        dtype = np.uint8
    elif dtype == np.int16:
        original_data = original_data + 128
        dtype = np.uint16
    original_data = original_data.astype(dtype)
    original_data = rearrange(original_data, "W H D T -> T D () H W")
    tifffile.imwrite(
        opj(opd(data_path), data_name + ".tif"),
        original_data,
        imagej=True,
    )
