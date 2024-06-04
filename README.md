# JPSTS-Net: Joint-wise Temporal Self-similarity Periodic Selection Network for Repetitive Fitness Action Counting

## Introduction

Pytorch implementation of "Joint-wise Temporal Self-similarity Periodic Selection Network for Repetitive Fitness Action Counting"
- paper download: https://ieeexplore.ieee.org/document/10534255

![Demo GIF](./figs/JTSPS-Net.gif)


# Usage
## 1. Dependencies
This code is tested on [Ubuntu 20.04 LTS，python 3.7，pytorch 1.8.1]. 
```
1. conda activate [your_enviroment]
2. pip install -r requirments.txt
```

## 2. Data
To fill the action counting gap in real physical fitness scenarios and to scale up the existing repetition counting datasets, we introduce a repetitive action counting dataset called FitnessRep in which all samples were collected in real physical fitness scenarios. We have released the  [FitnessRep](https://1drv.ms/f/s!AgILGCFlb65Qj1KEsq9PD7V18QBy?e=9F9LT8) dataset.

## 3. Test
Place the [pretrained model](https://1drv.ms/u/s!AgILGCFlb65Qj1EJqlfJHWk_ogp1?e=fgq3K4) at `checkpoint/` and run
```
python test.py 
```
## Citation

If you find our work useful in your research, please consider citing:

```
@ARTICLE{10534255,
  author={Huang, Hu and Gou, Shuiping and Li, Ruimin and Gao, Xinbo},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Joint-wise Temporal Self-similarity Periodic Selection Network for Repetitive Fitness Action Counting}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2024.3402728}}

```

