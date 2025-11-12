# CoVaR Estimation for Systemic Risk Using MCMC

This directory contains two complete implementations of CoVaR (Conditional Value-at-Risk) estimation using different MCMC approaches.

## What is CoVaR?

**CoVaR** measures how much the financial system's risk increases when a specific institution is in distress. It's a key metric for identifying systemically important financial institutions ("too big to fail").

**ΔCoVaR** = CoVaR - VaR_system

This quantifies the institution's contribution to systemic risk.

## Implementations

### 1. Custom Gibbs Sampler (`covar_mcmc_estimation.py`)
- **Method**: Manual Gibbs sampling with data augmentation
- **Purpose**: Educational - shows MCMC mechanics explicitly
- **Algorithm**: Metropolis-within-Gibbs for quantile regression
- **Speed**: ~50 seconds per model
- **Code**: ~600 lines, fully self-contained

**Run it:**
```bash
python covar_mcmc_estimation.py
```

**Outputs:**
- `covar_mcmc_results.png` - 6-panel visualization
- `covar_results.csv` - Numerical results with credible intervals

### 2. PyMC with NUTS (`covar_pymc_estimation.py`)
- **Method**: Modern probabilistic programming
- **Purpose**: Production-ready implementation
- **Algorithm**: No-U-Turn Sampler (advanced Hamiltonian Monte Carlo)
- **Speed**: ~2 seconds per model (25x faster!)
- **Code**: ~450 lines with automatic diagnostics

**Run it:**
```bash
python covar_pymc_estimation.py
```

**Outputs:**
- `covar_pymc_results.png` - 12-panel visualization with diagnostics
- `covar_pymc_results.csv` - Results with convergence metrics

## Key Results

Both implementations successfully:
- ✅ Estimate institution and system VaR using Bayesian quantile regression
- ✅ Compute CoVaR through quantile regression at tail level
- ✅ Calculate ΔCoVaR (systemic risk contribution)
- ✅ Provide full uncertainty quantification (95% credible intervals)
- ✅ Generate comprehensive visualizations

## The Methodology (Adrian-Brunnermeier 2016)

```
Step 1: Estimate institution's VaR (q_α^i)
Step 2: Estimate system's unconditional VaR (q_α^system)
Step 3: Run quantile regression:
        system_t = α + β × institution_t + γ × system_{t-1} + ε_t
Step 4: CoVaR_α^i = α + β × q_α^i + γ × median(system_{t-1})
Step 5: ΔCoVaR = CoVaR_α^i - q_α^system
```

## Technical Details

### Bayesian Quantile Regression
- Uses Asymmetric Laplace Distribution (ALD) likelihood
- The τ-quantile corresponds to the mode of ALD
- MCMC samples from posterior: p(θ|data) ∝ p(data|θ) × p(θ)

### MCMC Algorithms

**Gibbs Sampling:**
- Updates parameters one at a time
- Uses data augmentation with latent scale variables
- Requires manual tuning and diagnostics

**NUTS (No-U-Turn Sampler):**
- Updates all parameters jointly
- Uses Hamiltonian dynamics with gradients
- Automatic tuning of step size and trajectory length
- Built-in convergence diagnostics (R-hat, ESS)

## Files in This Directory

| File | Description |
|------|-------------|
| `covar_mcmc_estimation.py` | Gibbs sampling implementation |
| `covar_pymc_estimation.py` | PyMC/NUTS implementation |
| `covar_mcmc_results.png` | Gibbs visualizations |
| `covar_pymc_results.png` | PyMC visualizations |
| `covar_results.csv` | Gibbs numerical results |
| `covar_pymc_results.csv` | PyMC numerical results |
| `COMPARISON_Gibbs_vs_PyMC.md` | Detailed comparison |
| `README.md` | This file |

## Requirements

**Gibbs Implementation:**
```
numpy
scipy
pandas
matplotlib
seaborn
tqdm
```

**PyMC Implementation:**
```
pymc
arviz
numpy
scipy
pandas
matplotlib
seaborn
```

## Real-World Applications

CoVaR is used by:
- **Central Banks** - For macroprudential supervision and stress testing
- **Regulators** - To identify systemically important institutions (SIFIs)
- **Risk Managers** - To quantify interconnectedness and contagion risk
- **Academic Research** - To study financial stability and network effects

## References

1. Adrian, T., & Brunnermeier, M. K. (2016). CoVaR. *American Economic Review*, 106(7), 1705-1741.

2. Koenker, R., & Bassett Jr, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.

3. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593-1623.

4. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. *Handbook of Markov Chain Monte Carlo*, 2(11), 2.

## Learning Path

1. **Start here**: Run `covar_mcmc_estimation.py` and study the code
2. **Understand**: Read each step of the Gibbs sampler
3. **Compare**: Run `covar_pymc_estimation.py` to see the difference
4. **Deep dive**: Read `COMPARISON_Gibbs_vs_PyMC.md`
5. **Apply**: Modify for your own financial data

## Author Notes

These implementations demonstrate how MCMC enables Bayesian inference in complex financial models where:
- Analytical solutions don't exist
- Traditional MLE struggles with tail estimation
- Uncertainty quantification is critical for risk management

The transition from manual Gibbs sampling to modern probabilistic programming (PyMC/Stan) mirrors the historical development in computational statistics, where tools developed in physics (Metropolis, Hamiltonian MC) were refined by statisticians (Neal, Rosenthal) for widespread practical use.

---

*For questions or issues, check the inline documentation in each script.*
