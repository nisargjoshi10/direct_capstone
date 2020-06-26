# In Silico Plasticizer Discovery
This package contains python modules for the analysis, prediction and discovery of unique, novel plasticizer molecules. We provide methods and classes for automatically generating molecular features, building a VAE, modifying its architecture and training it on a GPU. The generative framework herein is not just limited to plasticizers and can also be optimized based on any property of your choice.

# Installation
`### TODO: MAKE PIP INSTALLABLE AND INTEGRATE WITH TRAVIS CI`

# Overview
The discovery of new materials has been at the forefront of many seminal scientific achievements and there exists a strong and active community of researchers who continue to search for more advanced, efficient and environmentally-benign materials. This includes finding a non-toxic and cheap alternative to [phthalates](https://www.theguardian.com/lifeandstyle/2015/feb/10/phthalates-plastics-chemicals-research-analysis), additives that are commonly used to increase the flexibility of plastics such as PVC. To aid in the discovery of new materials for a given application, researchers have recently begun to implement an "inverse-design" paradigm, in which molecular structure is derived from a desired property value rather than vice versa. This enables us to search broadly through molecular phase space and also helps us to expand our search beyond traditional chemical intuition. Many generative algorithms have been used for this purpose, but in this study we focus specifically on the [variational autoencoder](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/). To generate new plasticizer candidates, we first derive a property that reflects the likelihood a given molecule will be a good plasticizer and then encode that property into the latent space of a VAE which can be optimized and used to generate brand new molecular structures.

## Plasticizer Likelihood
