# Implicit Neural Compression (INC)
Welcome to INC! This repository is aimed at helping researchers or developers interested in
INC data compression algorithms to quickly understand and reproduce our latest research.

Implicit Neural Compression (INC) refers to the data copmpression methods that represent
the target discrete grid-based data with continuous implicit functions parameterized by neural networks.

The source code of SCI will be released soon!

# SCI: A spectrum concentrated implicit neural compression for biomedical data

## Quickstart

### 1. Create a conda environment

    conda create -n sci python=3.10
    conda activate sci

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

### 3. Register Gurobi and get license

[Register and install an academic license](https://www.gurobi.com/downloads/free-academic-license/) 
for the Gurobi optimizer (this is free for academic use).

### 4. Compression

(1) Compress small data without any partitioning strategy.

relevant compression parameters can be modified in **opt/SingleTask/default.yaml**.

    bash 1_Single.sh
final compressed file path: **outputs/single_{time}/exp/steps{step}/compressed/**

final decompressed file path: **outputs/single_{time}/exp/steps{step}/decompressed/{name}_decompressed.tif**

training result: 

    tensorboard --logdir=outputs/single_{time}

(2) Compress big data with adaptive partitioning.

relevant compression parameters can be modified in **opt/DivideTask/default.yaml**.

    bash 2_Divide.sh
final compressed file path: **outputs/divide_{time}/exp/steps{step}/compressed/**

final decompressed file path: **outputs/divide_{time}/exp/steps{step}/decompressed/{name}_decompressed.tif**

training result: 

    tensorboard --logdir=outputs/divide_{time}

(3) Run multiple tasks at once.

relevant compression parameters can be modified in **opt/MultiTask/default.yaml**.

    bash 3_MultiExp.sh
final compressed file path: **outputs/{project}_{time}/exp/steps{step}/compressed/**

final decompressed file path: **outputs/{project}_{time}/exp/steps{step}/decompressed/{name}_decompressed.tif**

training result: 

    tensorboard --logdir=outputs/{project}_{time}




