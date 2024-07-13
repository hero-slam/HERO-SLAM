# HERO-SLAM: Hybrid Enhanced Robust Optimization of Neural SLAM
This is an official implementation of our work published in ICRA'24. [Project Page](https://hero-slam.github.io/)

> **HERO-SLAM: Hybrid Enhanced Robust Optimization of Neural SLAM**
>
> Zhe Xin<sup>1</sup>,  Yufeng Yue<sup>2</sup>, Liangjun Zhang<sup>3</sup> and Chenming Wu<sup>3</sup><br>
> <sup>1</sup>Meituan UAV, Beijing, China,
> <sup>2</sup>School of Automation, Beijing Institute of Technology,
> <sup>3</sup>Robotics and Autonomous Driving Lab (RAL), Baidu Research
> 
> [**Paper** (arXiv)](https://hero-slam.github.io/)


## Installation

Please follow the instructions below to install the repo and dependencies.

```bash
git clone https://github.com/hero-slam/HERO-SLAM
cd HERO-SLAM
```



### Install the environment

```bash
# Create conda environment
conda create -n heroslam python=3.7
conda activate heroslam

# Install the pytorch first (Please check the cuda version)
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -r requirements.txt

# Build extension (marching cubes from neuralRGBD)
cd external/NumpyMarchingCubes
python setup.py install

```

## Dataset

Please follow the instructions provided here: https://github.com/HengyiWang/Co-SLAM for downloading and preprocessing datasets.


## Run

You can run HERO-SLAM using the code below.

For fixd interval, controlled by "ds_interval" in config file

```
python heroslam_opt_pose.py --config './configs/{Dataset}/{scene}.yaml 
```

For adaptive interval, controlled by "run_interval" in config file, and "ds_interval = 1" in this mode.

```
python heroslam_opt_pose_adaptive_interval.py --config './configs/{Dataset}/{scene}.yaml 
```



## Evaluation

Code for evaluation strategy, performance analysis [click here](https://github.com/JingwenWang95/neural_slam_eval).



## Acknowledgement

Our codebase builds on the code in [Co-SLAM](https://github.com/HengyiWang/Co-SLAM).

## Citation

If you find our code or paper useful for your research, please consider citing:

```
@inproceedings{xin2024heroslam,
        title={HERO-SLAM: Hybrid Enhanced Robust Optimization of Neural SLAM},
        author={Zhe Xin, Yufeng Yue, Liangjun Zhang and Chenming Wu},
        booktitle={ICRA},
        year={2024}
}
```