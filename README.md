# public-adaptive-sampling-dendritic
Repository with all the code related to https://arxiv.org/abs/2603.00093 article, now under revisions in Journal of Physics: Materials.

## Description
The high computational cost of phase field simulations remains a major limitation for predicting dendritic solidification in metals, particularly in additive manufacturing, where microstructural control is critical. This work investigates alternative strategies for surrogate modeling of dendritic solidification using uncertainty-driven adaptive sampling with XGBoost and CNNs, including a self-supervised strategy, to efficiently approximate the spatio-temporal evolution while reducing costly phase field simulations. The proposed adaptive strategy leverages model uncertainty, approximated via Monte Carlo dropout for CNNs and bagging for XGBoost, to identify high-uncertainty regions where new samples are generated locally within hyperspheres, progressively refining the spatio-temporal design space and achieving accurate predictions with significantly fewer phase field simulations than an Optimal Latin Hypercube Sampling optimized via discrete Particle Swarm Optimization (OLHS-PSO). The framework systematically investigates how temporal instance selection, adaptive sampling, and the choice between domain-informed and data-driven surrogates affect spatio-temporal model performance. Evaluation considers not only computational cost but also the number of expensive phase field simulations, surrogate accuracy, and associated CO2 emissions, providing a comprehensive assessment of model performance as well as their related environmental impact.

## Requirements
Octave version 9.3.0
Python environment is managed using conda/miniconda. 

## Instalation
### Clone the dependencies
```bash
git clone https://github.com/eidergarate/public-adaptive-sampling-dendritic
cd public-adaptive-sampling-dendritic
```
### Install python repositories
For windows:
```bash
conda env create -f win-env.yml
```
For linux:
```bash
conda env create -f linux-env.yml
```
### NOTE
In order to execute the physical model you need to copy octave-cli path to physical-model/runMatlab.py.

## Getting started
For reproducing article results you need to execute adaptive-sampling/main_test_ini_all_t.py and classical_sampling/main_exp_ini_all_t.py. 

These functions call to the sampling methods, execute corresponding physical model's experiments, train the 3 types of models for the 3 temporal instances sets and all sample sizes. When finilized, all results are stored and can be plotted using results_code python scripts.

## Contact
- **Name:** Eider Garate Perez
- **Email:** eider.garate@tekniker.es




