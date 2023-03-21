# 💾Implicit Neural Compression (INC)
Our paper was accepted to AAAI2023. You can also find our full-version paper [on arXiv](https://arxiv.org/abs/2209.15180)

<img src="docs/assets_readme/SCI_method.jpg" width="80%"/>

<img src="docs/assets_readme/SCI_compare_roi.jpg" width="80%"/>


# 🚀Quickstart

### 0. Clone repository

	git clone git@github.com:RichealYoung/ImplicitNeuralCompression.git
	cd ImplicitNeuralCompression
### 1. Create a conda environment

    conda create -n inc python=3.10
    conda activate inc

### 2. Install python libraries
	pip3 install -r requirements.txt
<!-- ### 3. Register Gurobi and get license

[Register and install an academic license](https://www.gurobi.com/downloads/free-academic-license/) 
for the Gurobi optimizer (this is free for academic use). -->

### 3. Compression

	python sci.py -c config/SingleExp/sci.yaml -g 2

All hyper-parameters can be set in the YAML file.

❗Note: The partition methods will be released soon!

# 🧰Batch Experiments
We have also provided a useful script 'BatchExp.py' for researchers to perform batch experiments quickly. You just need to configure this group of experiments in the YAML file, pick the GPUs, and start batch experiments with the following command.

	python BatchExp.py -c config/BatchExp/sci.yaml -stp sci.py -g 0,1,2,3

These experiments will automatically wait or execute depending on GPU utilization..
# 😘Citations

	@misc{yang2022sci,
      title={SCI: A Spectrum Concentrated Implicit Neural Compression for Biomedical Data}, 
      author={Runzhao Yang and Tingxiong Xiao and Yuxiao Cheng and Qianni Cao and Jinyuan Qu and Jinli Suo and Qionghai Dai},
      year={2022},
      eprint={2209.15180},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
	}

# 💡Contact
If you need any help or are looking for cooperation feel free to contact us.
yangrz20@mails.tsinghua.edu.cn
