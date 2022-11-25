# 💾Implicit Neural Compression (INC)
Welcome to INC! This repository is aimed at helping researchers or developers interested in
INC data compression algorithms to quickly understand and reproduce our latest research.

<img src="docs/assets_readme/SCI_method.jpg" width="100%"/>

<img src="docs/assets_readme/SCI_compare_roi.jpg" width="80%"/>


# 🚀Quickstart

### 1. Create a conda environment

    conda create -n inc python=3.10
    conda activate inc

### 2. Install python libraries.
	conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
    pip3 install omegaconf
	pip3 install pynvml
	pip3 install psutil
	pip3 install tqdm
	pip3 install matplotlib
	pip3 install pandas
	pip3 install einops
	pip3 install tifffile
	pip3 install opencv-python
	pip3 install scipy
	pip3 install gurobipy
	pip3 install tensorboard
    pip3 install scikit-image

<!-- ### 3. Register Gurobi and get license

[Register and install an academic license](https://www.gurobi.com/downloads/free-academic-license/) 
for the Gurobi optimizer (this is free for academic use). -->

### 3. Compression

	python main.py -c config/SingleTask/default.yaml -g 2

All hyper-parameters can be set in the YAML file.

❗Note: The partition methods will be released soon!

# 😘Citations

	@misc{https://doi.org/10.48550/arxiv.2209.15180,
	doi = {10.48550/ARXIV.2209.15180},
	
	url = {https://arxiv.org/abs/2209.15180},
	
	author = {Yang, Runzhao and Xiao, Tingxiong and Cheng, Yuxiao and Cao, Qianni and Qu, Jinyuan and Suo, Jinli and Dai, Qionghai},
	
	title = {SCI: A Spectrum Concentrated Implicit Neural Compression for Biomedical Data},
	
	publisher = {arXiv},
	
	year = {2022},
	
	copyright = {arXiv.org perpetual, non-exclusive license}
	}

# 😀Contact
If you need any help or are looking for cooperation feel free to contact us.
yangrz20@mails.tsinghua.edu.cn