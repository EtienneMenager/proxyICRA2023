# Toward the use of proxies for efficient learning manipulation and locomotion strategies on soft robots.

[![SOFA](https://img.shields.io/badge/SOFA-on_github-orange.svg)](https://github.com/SofaDefrost/sofa)

This file contains the code used to obtain the results of the article "Toward the use of proxies for efficient learning manipulation and locomotion strategies on soft robots" published in RAL paper in 2023.  

# Quick start

## How to use

Please integrate the various elements of this folder into the [SofaGym](https://github.com/SofaDefrost/SofaGym) framework. The various components work with version v22.12 of SofaGym. The proposed paths do not necessarily correspond to the paths in your installation.

## Folders

This folder is divided into 3 subfolders:

* Envs: Gym environment describing proxy and FEM models.
* Optimization: Script for optimising proxy parameters using Bayesian methods.
* Transfer: script for managing both gym environments and transfer to a Sofa scene.

# Citing
If you use the project in your work, please consider citing it with:

```bibtex
@article{menager2023toward,
  title={Toward the use of proxies for efficient learning manipulation and locomotion strategies on soft robots},
  author={M{\'e}nager, Etienne and Peyron, Quentin and Duriez, Christian},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
```
