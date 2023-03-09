# Contrastive Hybrid Attention Vision Transformer (HA-ViT)

### Introduction
This repository contains the source code for the paper [Face-Periocular Cross-Identification via Contrastive Hybrid Attention Vision Transformer], which is accepted by IEEE Signal Processing Letters.


### Dataset
We use the Cross-modal Face-periocular Dataset to train the proposed model, which are available as follows:
- [Cross-modal Face-periocular](https://www.kaggle.com/datasets/leslietiong/cmfpdb)


### Compatibility
We tested the codes with:
  1) PyTorch 1.12.0 with and without GPU under Ubuntu 18.04 and Anaconda3 (Python 3.8 and above)
  2) PyTorch 1.10.2 with and without GPU under Windows 10 and Anaconda3 (Python 3.7 and above)
  

### Requirements
  1) [Anaconda3](https://www.anaconda.com/distribution/#download-section)
  2) [PyTorch](https://pytorch.org/get-started/locally/)
  3) [Natsort](https://pypi.org/project/natsort/)


### Usage
- Run the code `main.py` with the given configuration in config.py
```shell
$ python main.py --training_mode True --pretrain_mode False --dataset_mode 'CMFP'
```
- Evaluate the model with the given test sample
```shell
$ python main.py --training_mode False --dataset_mode 'other'
```


### Pretrained Model
The pretrained model will be shared soon.


### License
This work is an open-source under MIT license.


### Cite this work
```
@article{HAViT2023,
    author    = {Tiong, Leslie Ching Ow and Sigmund, Dick and Teoh, Andrew Beng Jin},
    title     = {Face-Periocular Cross-Identification via Contrastive Hybrid Attention Vision Transformer},
    year      = {2023}
}
```
