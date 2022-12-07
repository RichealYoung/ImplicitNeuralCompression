from einops import rearrange
import tifffile
import nibabel
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops

if __name__ == "__main__":
    data_path = "/ssd/0/yrz/ffmpeg/dataset/brain-fmri.nii.gz"
    data_name = opb(data_path).replace(".nii.gz", "")
    original_data_ = nibabel.load(data_path)
    original_data = original_data_.get_fdata().astype(original_data_.get_data_dtype())
    original_data = rearrange(original_data, "W H D T -> T D () H W")
    tifffile.imwrite(
        opj(opd(data_path), data_name + ".tif"),
        original_data,
        imagej=True,
    )
