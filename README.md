# DSR-EI
This repo is the PyTorch implementation of our paper in CVIU-2023. The algorithm can be used to perform DSR-EI, and its architecture is

![figure](imgs/architecture.png)

## Checkpoints
All pre-trained models can be found [here](https://drive.google.com/drive/folders/1nu2xcxpHrfid1tUjplqwYw1hz-yRFfa0?usp=sharing).

## Datasets
### NYUv2
We use a [preprocessed version](https://drive.google.com/drive/folders/1_1HpmoCsshNCMQdXhSNOq8Y-deIDcbKS?usp=sharing) provided [here](https://github.com/charlesCXK/RGBD_Semantic_Segmentation_PyTorch#data-preparation).
### RGBDD
We follow [FDSR](https://openaccess.thecvf.com/content/CVPR2021/papers/He_Towards_Fast_and_Accurate_Real-World_Depth_Super-Resolution_Benchmark_Dataset_and_CVPR_2021_paper.pdf) and use the data provided [here](https://github.com/lingzhi96/RGB-D-D-Dataset) (need to assign release agreement).
### DIML
Download the indoor data from [here](https://dimlrgbd.github.io) and extract it into `./data/DIML/{train,test}` respectively. Then following [LGR](https://github.com/prs-eth/graph-super-resolution), run `python scripts/create_diml_npy.py ./data/DIML` to create numpy binary files for faster data loading.
### Middlebury
Following [LGR](https://raw.githubusercontent.com/prs-eth/graph-super-resolution/master/README.md), download the dataset [here](https://vision.middlebury.edu/stereo/data/) and place the extracted scenes in `./data/Middlebury/<year>/<scene>`. For the 2005 dataset, make sure to only put the scenes for which ground truth is available. The data splits are defined in code.

## Dependencies
- Python >= 3.8
- [PyTorch](https://pytorch.org/) >= 1.11
- [Cosine Annealing with Warmup for PyTorch](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py)
- [einops](https://einops.rocks)
- [mmedit](https://pypi.org/project/mmedit/)
- [timm](https://timm.fast.ai)
- tensorboardX, tqdm, torchsummary, Pillow, pathlib

## Train and Test (take $4\times$ as an example)
Specify your own 'dataset_path', 'dataset', 'data_root', 'epoch' and the corresponding 'first_cycle_steps' in main.py. Then train or test the model using the following code:
> python ./main.py --scale 4 --scratch

> python ./main.py --scale 4 --test --best

## Citation

```
@article{qiao2023depth,
title = {Depth super-resolution from explicit and implicit high-frequency features},
journal = {Computer Vision and Image Understanding},
volume = {237},
pages = {103841},
year = {2023},
issn = {1077-3142},
author = {Xin Qiao and Chenyang Ge and Youmin Zhang and Yanhui Zhou and Fabio Tosi and Matteo Poggi and Stefano Mattoccia}}
```
