# Learning Multimodal Graph-to-Graph Translation for Molecular Optimization

This is the official implementation of junction tree encoder-decoder model in https://arxiv.org/abs/1812.01070

## Requirements
* Python == 2.7
* RDKit >= 2017.09
* PyTorch >= 0.4.0
* Numpy
* scikit-learn

The code has been tested under python 2.7 with pytorch 0.4.1.

## Quick Start
The tutorial of training and testing our variational junction tree encoder-decoder is in [diff_vae/README.md](./diff_vae).

A quick summary of different folders:
* `data_prep/` contains cosine, Jaccard, and Tanimoto similarity preprocessing scripts.
* `diff_vae/` includes the training and decoding script of variational junction tree encoder-decoder ([README](./diff_vae)).
* `diff_vae_gan/` includes the training and decoding script of adversarial training module ([README](./diff_vae_gan)).
* `fast_jtnn/` contains the implementation of junction tree encoder-decoder.
* `props/` is the property evaluation module, including penalized logP, QED and DRD2 property calculation.
* `scripts/` provides model evaluation, visualization and preprocessing scripts.

## Contact
Wengong Jin (wengong@csail.mit.edu)
