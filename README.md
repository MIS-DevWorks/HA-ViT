# Contrastive Hybrid Attention Vision Transformer (HA-ViT)

### Introduction
This repository contains the source code for one of the main contributions in this paper [Face-Periocular Cross-Identification via Contrastive Hybrid Attention Vision Transformer](https://ieeexplore.ieee.org/document/10068230), which is accepted by IEEE Signal Processing Letters.


### Dataset
We use the Cross-modal Face-periocular Dataset to train the proposed model, which are available as follows:
- [Cross-modal Face-periocular](https://www.kaggle.com/datasets/leslietiong/cmfpdb)


### Requirements
  1) [Anaconda3](https://www.anaconda.com/download)
  2) [PyTorch](https://pytorch.org/get-started/locally)
  3) [Natsort](https://pypi.org/project/natsort)


### Usage
- Run the code `main.py` with the given configuration in config.py
```shell
$ python main.py --training_mode --dataset_mode 'CMFP'
```
- Evaluate the model with the given test sample (see Sample Images.zip)
```shell
$ python main.py --dataset_mode 'other'
```

### Compatibility
We tested the codes with:
  1) PyTorch &ge; 1.12.0 with and without GPU under Ubuntu 18.04 and Anaconda3 (Python 3.8 and above)
  2) PyTorch &ge; 1.10.2 with and without GPU under Windows 10 and Anaconda3 (Python 3.7 and above)


### Pretrained Model
The pretrained model can be found [here](https://drive.google.com/file/d/1N9US24HqEc1VypsILlJd8slX_fEP2rlK/view?usp=share_link).


### License
This work is an open-source under MIT license.


### Cite this work
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
