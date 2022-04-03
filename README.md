# SI-Dial
This is the Pytorch implementation for the paper [Supplementing Missing Visions via Dialog for Scene Graph Generations]().

## 1. Project Overview
In this work, we aim to explore the Scene Graph Generation (SGG) task under the setting of insufficient visual input, and propose to supplement the missing visions via dialog.

Our implementations are based on the [codebase](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), we want to thank the original authors for sharing their code. 

<p align="center">
	<img src="assets/Figure1.png" width="500">

## 2. Environment Setup
The environment setup requirements are in general the same as [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/INSTALL.md) and [Maskrcnn-Benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Please follow the installation guidance.

## 3. Data Preparation
In this work, we explore the SGG task with insufficient visual input. Therefore, instead of using the original images, we first pre-processing the VG dataset to obtain three levels of vision missingness: 



