# Bayesian Motif Detection via MCMC & Gibbs Sampling
### Discrete-Time Markov Chains, Dirichlet-Multinomial Conjugacy, and Metropolis-Hastings Phase Shifts

![Language](https://img.shields.io/badge/Language-Python-blue)
![Domain](https://img.shields.io/badge/Domain-Stochastic%20Processes%20%7C%20MCMC-orange)
![Methods](https://img.shields.io/badge/Methods-Gibbs%20Sampling%20%7C%20Metropolis--Hastings-green)
![Course](https://img.shields.io/badge/Course-MATH1222%20%40%20ULi%C3%A8ge-red)

---

## Executive Summary

This repository hosts a production-grade probabilistic framework designed to model sequential discrete processes and discover recurring patterns (*motifs*) hidden within noisy background sequences. Developed for the **MATH1222: Introduction to Stochastic Processes** course at the University of Liège, this project combines mathematical derivations of Markov chains with advanced Markov Chain Monte Carlo (MCMC) inference.

### Key Scientific & Engineering Highlights
- **Discrete-Time Markov Chains (DTMC)**: Formulated maximum likelihood estimators (MLE) via constrained Lagrangian optimization to estimate transition probability matrices $P$. Analytically and numerically established the stationary distribution $\pi_\infty P = \pi_\infty$ using spectral powers and steady-state balance systems.
- **Bayesian Modeling & Conjugacy**: Modeled motif positions and signatures using a **Dirichlet-Multinomial conjugate prior**, enabling closed-form conditional posterior sampling for the emission profile $\Theta$.
- **Markovian Background Architecture**: Extended background modeling from independent categorical variables to 1st-order Markov chains $\Phi = (u_\Phi, P_\Phi)$, capturing temporal dependencies across flanking regions.
- **MCMC Sampling with Metropolis-Hastings Shifts**: Implemented a randomized Gibbs sampler enhanced with periodic Metropolis-Hastings shift proposals ($\delta \in \{-2, -1, 1, 2\}$) to avoid trapping in local probability extrema.
- **Vectorized High-Performance Python**: Vectorized likelihood updates and state sampling using NumPy array broadcasting, achieving an **execution speedup factor of $\approx 50\times$** over naive nested loops.
- **Competitive Benchmarks**: Ranked **3rd place** in both synthetic ($AJI = 0.80, PPV = 0.80$) and real biological benchmark datasets ($AJI = 0.70, PPV = 0.89$).

---

## Repository Structure

```text
├── code                    # Code for the second part of the project
├── README.md               # Engineering report & mathematical documentation
├── report.pdf              # Report in french
└── statement               # Guidelines of the project   