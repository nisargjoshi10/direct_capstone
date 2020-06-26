# In Silico Plasticizer Discovery
This package contains python modules for the analysis, prediction and discovery of unique, novel plasticizer molecules. We provide methods and classes for automatically generating molecular features, building a VAE, modifying its architecture and training it on a GPU. The generative framework herein is not just limited to plasticizers and can also be optimized based on any property of your choice.

# Installation
`### TODO: MAKE PIP INSTALLABLE AND INTEGRATE WITH TRAVIS CI ###`

`### TODO: COMMENT FUNCTIONS AND ADD FULL DOCUMENTATION ###`

# Overview
The discovery of new materials has been at the forefront of many seminal scientific achievements and there exists a strong and active community of researchers who continue to search for more advanced, efficient and environmentally-benign materials. This includes finding a non-toxic and cheap alternative to [phthalates](https://www.theguardian.com/lifeandstyle/2015/feb/10/phthalates-plastics-chemicals-research-analysis), additives that are commonly used to increase the flexibility of plastics such as PVC. To aid in the discovery of new materials for a given application, researchers have recently begun to implement an "inverse-design" paradigm, in which molecular structure is derived from a desired property value rather than vice versa. This enables us to search broadly through molecular phase space and also helps us to expand our search beyond traditional chemical intuition. Many generative algorithms have been used for this purpose, but in this study we focus specifically on the [variational autoencoder](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/).

![VAE Diagram](/docs/readme_figs/vae_diagram.png)

To generate new plasticizer candidates, we first derive a property that reflects the likelihood a given molecule will be a good plasticizer and then encode that property into the latent space of a VAE which can be optimized and used to generate brand new molecular structures.

## Plasticizer Likelihood
There is currently no single property in existence that is fully descriptive of the plasticizing capabilities of a given molecule. Molecular weight, volatility, logP and polarity are all important indicators, however when attempting a large-scale search across thousands of molecules these properties are not unique enough on their own to adequately separate likely from non-likely plasticizers. To derive a property that meets our requirements we define a multi-level filter that allows us to combine a coarse layer that removes obvious non-candidates with a fine layer that accounts for differences in quality between already known plasticizers. We achieve this by:
1. Generating 195 features for each molecule using the RDKit python package
2. Hand selecting a set of non-plasticizing molecules to use as negative samples for training a logistic regression model (coarse)
3. Fitting a linear regression model to a proprietary ordinal ranking of plasticizers provided by Cargill, Inc. (fine)
4. Combining the results from the coarse and fine models to obtain a single value used as a proxy for the plasticizing effectiveness of a given molecule.

`### TODO: IMPROVE SELECTION OF NON-PLASTICIZERS ###`
`### TODO: IMPLEMENT FINE MODEL ###`

Features for any list of smiles can be generated using `feat_df = gen_features(smiles)`. The `likelihood_predictor` module contains the classes necessary for building and training the coarse and fine models as well as combining them into a single property. An example of the full training pipeline is provided in the `notebooks` folder.

`### TODO: ADD NOTEBOOK TUTORIAL FOR FULL PREDICTIVE PIPELINE ###`

## Generating New Candidates
The `vae_generator` module can be used to build and train a VAE for any set of smiles and property. Models pretrained on a subset of the GDB-17 dataset are available in `modules/checkpoints` (NOTE: these models were trained using the GRUGRU architecture). You can choose between convolutional layers, GRU layers or biGRU layers for the encoder and GRU layers or biGRU layers for the decoder. You can also add your property of choice by passing a 2D array with the SMILES and property to the training method. Examples of model creation and training can also be found in `notebooks`.

`### TODO: ADD NOTEBOOK TUTORIAL FOR FULL GENERATIVE PIPELINE ###`
