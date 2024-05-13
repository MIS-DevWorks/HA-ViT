<h1 align="center">
  Face-Periocular Cross-Identification via Contrastive Hybrid Attention Vision Transformer
</h1>
<p align="center">
  Leslie Ching Ow Tiong<sup>1</sup>,  Dick Sigmund<sup>2</sup>, Andrew Beng Jin Teoh<sup>&dagger;,3</sup>
  <br/>
  <sup>1</sup>Korea Institute of Science and Technology, <sup>2</sup>AIDOT Inc., <sup>3</sup>Yonsei University
  <br/>
  <sup>&dagger;</sup>Corresponding author
  <br/><br/>
  <a href="https://ieeexplore.ieee.org/document/10068230">
    <img src="https://img.shields.io/badge/paper-link-blue.svg" alt="Paper Link">
  </a>
</p>
<br/> <br/>


## Introduction
This repository contains the source code for one of the main contributions in this paper [Face-Periocular Cross-Identification via Contrastive Hybrid Attention Vision Transformer](https://ieeexplore.ieee.org/document/10068230), which is accepted by **IEEE Signal Processing Letters**.
<br/> <br/>


## Dataset
We use the Cross-modal Face-periocular Dataset to train the proposed model, which are available as follows:
- [Cross-modal Face-periocular](https://www.kaggle.com/datasets/leslietiong/cmfpdb)
<br/> <br/>


## Requirements
  1) [Anaconda3](https://www.anaconda.com/download)
  2) [PyTorch](https://pytorch.org/get-started/locally)
  3) [Natsort](https://pypi.org/project/natsort)
<br/> <br/>


## Usage
- Run the code `main.py` with the given configuration in config.py
```shell
$ python main.py --training_mode --dataset_mode 'CMFP'
```
- Evaluate the model with the given test sample (see Sample Images.zip)
```shell
$ python main.py --dataset_mode 'other'
```
<br/> <br/>


## Compatibility
We tested the codes with:
  1) PyTorch &ge; 1.12.0 with and without GPU under Ubuntu 18.04 and Anaconda3 (Python 3.8 and above)
  2) PyTorch &ge; 1.10.2 with and without GPU under Windows 10 and Anaconda3 (Python 3.7 and above)
<br/> <br/>


## Pretrained Model
The pretrained model can be found [here](https://drive.google.com/drive/folders/1kRZWlPoNmC0JUR2IddKf0BjwXkOhyZ07?usp=sharing).
<br/> <br/>


## License
This work is an open-source under MIT license.
<br/> <br/>


## Cite this work
```
@article{HAViT2023,
    author    = {Tiong, Leslie Ching Ow and Sigmund, Dick and Teoh, Andrew Beng Jin},
    title     = {Face-Periocular Cross-Identification via Contrastive Hybrid Attention Vision Transformer},
    journal   = {IEEE Signal Processing Letters},
    doi       = {10.1109/LSP.2023.3256320},
    volume    = {30},
    pages     = {254--258},
    year      = {2023}
}
```
