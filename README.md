# Learning Multimodal Graph-to-Graph Translation for Polymer Optimization
This is an extension of the junction tree encoder-decoder model in https://arxiv.org/abs/1812.01070.

## Requirements
### Dependencies
* Python == 2.7
* RDKit >= 2017.09
* PyTorch >= 0.4.0
* Numpy
* scikit-learn
### Big Data Tools
* Scala >= 2.11.8

The code has been tested under Python 2.7 with PyTorch 0.4.1.

## Quick Start
The tutorial of training and testing the variational junction tree encoder-decoder is in [diff_vae/README.md](./diff_vae).

A quick summary of different folders:
* `data_prep/` contains Cosine, Jaccard, and Tanimoto similarity preprocessing scripts.
  * `DataPrep_Scala/` includes Scala implementations of Cosine & Jaccard similarity.
* `diff_vae/` includes the training and decoding script of variational junction tree encoder-decoder ([README](./diff_vae)).
* `diff_vae_gan/` includes the training and decoding script of adversarial training module ([README](./diff_vae_gan)).
* `fast_jtnn/` contains the implementation of junction tree encoder-decoder.
* `props/` is the property evaluation module, including penalized logP, QED and DRD2 property calculation.
* `scripts/` provides model evaluation, visualization and preprocessing scripts.

## Contact
| Name         | Email                     |
|--------------|---------------------------|
| Usman Ashraf | usman.ashraf678@gmail.com |
| Rishi Gurnani | rgurnani96@gatech.edu |
| Kenny Scharm | kscharm3@gatech.edu |
| Dan Tu | tudan0103@gmail.com |