# CMINNs: Compartment Model Informed Neural Networks - Unlocking Drug Dynamics


This repository contains the code and resources for the paper: [CMINNs: Compartment Model Informed Neural Networks - Unlocking Drug Dynamics](https://arxiv.org/abs/2409.12998). Our work introduces an innovative approach to pharmacokinetics (PK) and pharmacodynamics (PD) modeling, which enhances traditional multi-compartment models using fractional calculus and time-varying parameters.

## Overview

Pharmacokinetics and pharmacodynamics (PKPD) modeling is critical for understanding drug absorption, distribution, and its impact on biological systems, particularly in the drug development process. Traditional models, such as multi-compartment models, can sometimes overcomplicate the representation of drug dynamics in heterogeneous tissues. Our approach addresses these challenges by introducing fractional calculus and time-varying parameters, simplifying the model while capturing complex drug behaviors such as:

- Anomalous diffusion
- Drug trapping and escape in heterogeneous tissues
- Drug resistance, persistence, and tolerance in cancer therapies
- Distributed delayed responses in multi-dose administrations

This methodology is powered by **Physics-Informed Neural Networks (PINNs)** and **fractional Physics-Informed Neural Networks (fPINNs)**. By integrating ordinary differential equations (ODEs) with integer/fractional derivatives and neural networks, we optimize parameter estimation for time-varying, constant, or piecewise constant parameters. Our approach provides new insights into drug-effect dynamics and improves the prediction of drug behavior in complex biological systems.

## Key Features
- **JAX Framework**: The implementation utilizes the JAX library for efficient computation and automatic differentiation.
- **PINNs and fPINNs**: We employ both integer-order and fractional-order PINNs for solving ODEs in pharmacokinetics and pharmacodynamics.
- **Fractional Calculus**: Incorporation of fractional derivatives allows us to model anomalous diffusion, capturing complex drug dynamics such as drug trapping in tissues.
- **Explainable Drug Dynamics**: Our approach simplifies the system of equations while providing clear insights into drug absorption rates, cancer cell death mechanisms, and therapeutic resistance.
- **Improved Efficiency**: The proposed models offer a more computationally efficient framework with only two ODEs, without sacrificing accuracy in representing drug behavior.


If you use this code or our methodology in your research, please cite the paper:

```
@article{daryakenari2025cminns,
  title={CMINNs: Compartment model informed neural networks—Unlocking drug dynamics},
  author={Daryakenari, Nazanin Ahmadi and Wang, Shupeng and Karniadakis, George},
  journal={Computers in Biology and Medicine},
  volume={184},
  pages={109392},
  year={2025},
  publisher={Elsevier}
}

```
