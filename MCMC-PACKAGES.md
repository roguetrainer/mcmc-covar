MCMC methods like **Metropolis-Hastings** and **Hamiltonian Monte Carlo (HMC)** are implemented in several robust packages across the major statistical programming languages.

Here are the main MCMC packages for Python, R, and Julia, categorized by their primary MCMC engine:

---

## Python Packages

Python has become a powerhouse for MCMC thanks to libraries that integrate probabilistic programming with the deep learning ecosystem (like TensorFlow and PyTorch).

| Package | MCMC Engine / Purpose | Key Feature |
| :--- | :--- | :--- |
| **PyMC** (formerly PyMC3) | HMC (via NUTS), Metropolis-Hastings | High-level, user-friendly **Probabilistic Programming Language (PPL)**. Built on top of **Aesara** (a graph-based computation library). |
| **CmdStanPy** / **ArviZ** | **Stan** (via No-U-Turn Sampler/NUTS) | Provides Python interface to the highly efficient C++ **Stan** engine. ArviZ is the de facto library for MCMC diagnostics and visualization in Python. |
| **TensorFlow Probability (TFP)** | HMC (NUTS), Sequential Monte Carlo (SMC), etc. | Integrates Bayesian methods directly into TensorFlow for GPU acceleration, especially useful for **Bayesian Neural Networks (BNNs)**. |

---

## R Packages

R is the traditional home for many powerful statistical packages. The **Stan** ecosystem is particularly dominant here.

| Package | MCMC Engine / Purpose | Key Feature |
| :--- | :--- | :--- |
| **rstan** | **Stan** (via NUTS) | The primary R interface for Stan. Known for its speed and stability in sampling complex posterior distributions. |
| **MCMCglmm** | Metropolis-Hastings, Gibbs Sampling | Specialized for **Generalized Linear Mixed Models (GLMMs)** with a Bayesian approach. Excellent for complex ecology and biological data. |
| **coda** | MCMC Diagnostics | Not a sampler, but a critical analysis tool. Provides convergence diagnostics (e.g., Gelman-Rubin statistic) and plots for MCMC output. |

---

## Julia Packages

Julia is gaining popularity in the MCMC space due to its speed (often rivaling C++) and composability.

| Package | MCMC Engine / Purpose | Key Feature |
| :--- | :--- | :--- |
| **Turing.jl** | NUTS, Sequential Monte Carlo (SMC) | A complete PPL in Julia. Known for its modern design, speed, and extensive support for different inference algorithms. |
| **DynamicHMC.jl** | HMC (NUTS) | A low-level, highly optimized implementation of **Hamiltonian Monte Carlo** that provides extremely fast sampling, often used as the backend for other PPLs. |

---

## Key Takeaway

For most modern applications, packages that use the **No-U-Turn Sampler (NUTS)**—an advanced variation of **Hamiltonian Monte Carlo (HMC)**—are preferred due to their efficiency in sampling complex, high-dimensional spaces. This includes **Stan** (via CmdStanPy/rstan) and **PyMC**.
