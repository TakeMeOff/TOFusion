# TOFusion

This is official Pytorch implementation of "[TOFusion: Text-guided and object-aware infrared and visible image fusion](https://www.sciencedirect.com/science/article/abs/pii/S0031320326005352)"

## Network Architecture

![image](https://github.com/TakeMeOff/TOFusion/blob/main/fig/fig1.png)

The overall structure of network

## To train

run `train_lightning.py` to train TOFusion

## To Test

Run `test_lightning.py` to predict deblurring fused images.

## Fusion Results

![image](https://github.com/TakeMeOff/TOFusion/blob/main/fig/fig2.png)



#### If this work is helpful to you, please cite it as:

    @article{CHEN2026113569,
    title = {TOFusion: Text-guided and object-aware infrared and visible image fusion},
    journal = {Pattern Recognition},
    volume = {179},
    pages = {113569},
    year = {2026},
    issn = {0031-3203},
    doi = {https://doi.org/10.1016/j.patcog.2026.113569},
    author = {Jun Chen and Wei Yu and Zhuo Cheng and Xin Tian and Jiayi Ma},
    keywords = {Image fusion, Vision-language models, Multi-modality, Object-aware},
    }

