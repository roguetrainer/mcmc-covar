"""
CoVaR Estimation Using PyMC for Systemic Risk Measurement

This implementation uses PyMC (a modern probabilistic programming framework)
to estimate CoVaR through Bayesian quantile regression. PyMC uses the 
No-U-Turn Sampler (NUTS), an advanced variant of Hamiltonian Monte Carlo,
which is much more efficient than basic Gibbs sampling.

Key advantages of PyMC:
1. Automatic tuning of MCMC parameters
2. Built-in convergence diagnostics (R-hat, ESS)
3. Efficient gradient-based sampling (NUTS/HMC)
4. Clean probabilistic programming syntax
5. Excellent visualization with ArviZ

Reference: Adrian & Brunnermeier (2016), "CoVaR"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def asymmetric_laplace_logp(value, mu, tau, b):
    """
    Custom log-likelihood for Asymmetric Laplace Distribution (ALD)
    
    The ALD is key for quantile regression as its mode corresponds to
    the tau-quantile of the distribution.
    
    Parameters:
    -----------
    value : array-like
        Observed values (residuals)
    mu : array-like
        Location parameter (regression mean)
    tau : float
        Quantile level (e.g., 0.05 for 5% VaR)
    b : float
        Scale parameter
    """
    import pytensor.tensor as pt
    
    residuals = value - mu
    # ALD log-likelihood - use PyTensor operations
    log_lik = pt.log(tau * (1 - tau) / b)
    # Use pt.switch for conditional logic instead of boolean indexing
    indicator = pt.switch(residuals < 0, 1.0, 0.0)
    log_lik = log_lik - (residuals * (tau - indicator)) / b
    return pt.sum(log_lik)


class CoVaREstimatorPyMC:
    """
    Estimates CoVaR and ΔCoVaR using PyMC with Bayesian quantile regression
    """
    
    def __init__(self, system_returns, institution_returns, alpha=0.05):
        """
        Parameters:
        -----------
        system_returns : array-like
            Returns of the financial system (e.g., market index)
        institution_returns : array-like
            Returns of the institution being analyzed
        alpha : float
            Confidence level (e.g., 0.05 for 95% VaR)
        """
        self.system_returns = np.array(system_returns)
        self.institution_returns = np.array(institution_returns)
        self.alpha = alpha
        self.results = {}
        
    def estimate_var_simple(self, returns, tau):
        """
        Simple empirical VaR estimation for comparison
        """
        return np.quantile(returns, tau)
    
    def estimate_var_bayesian(self, returns, tau, draws=2000, tune=1000):
        """
        Estimate VaR using Bayesian quantile regression with PyMC
        
        Model: y_i ~ AsymmetricLaplace(mu, tau, b)
               where mu = beta_0 (intercept only)
        
        Parameters:
        -----------
        returns : array-like
            Return series
        tau : float
            Quantile level
        draws : int
            Number of posterior samples to draw
        tune : int
            Number of tuning/warmup samples
            
        Returns:
        --------
        var_estimate : float
            Posterior median of VaR
        trace : arviz InferenceData
            Full posterior samples
        """
        print(f"\n  Building Bayesian quantile regression model for τ={tau}...")
        
        with pm.Model() as var_model:
            # Priors
            # Intercept (location of quantile)
            beta_0 = pm.Normal('beta_0', mu=np.mean(returns), sigma=np.std(returns)*3)
            
            # Scale parameter for ALD (weakly informative)
            b = pm.HalfNormal('b', sigma=1.0)
            
            # Likelihood using custom distribution
            mu = beta_0
            
            # Use CustomDist for asymmetric Laplace
            y_obs = pm.CustomDist(
                'y_obs',
                mu, tau, b,
                logp=asymmetric_laplace_logp,
                observed=returns
            )
            
            # Sample from posterior using NUTS
            print(f"  Sampling with NUTS (draws={draws}, tune={tune})...")
            trace = pm.sample(
                draws=draws, 
                tune=tune, 
                cores=2,
                return_inferencedata=True,
                progressbar=True
            )
        
        # Extract VaR estimate (beta_0 is the tau-quantile)
        var_samples = trace.posterior['beta_0'].values.flatten()
        var_estimate = np.median(var_samples)
        
        return var_estimate, trace, var_samples
    
    def estimate_covar_bayesian(self, draws=2000, tune=1000, include_lagged=True):
        """
        Estimate CoVaR using PyMC with Bayesian quantile regression
        
        This implements the full Adrian-Brunnermeier methodology:
        1. Estimate institution's VaR
        2. Estimate system's unconditional VaR  
        3. Run quantile regression: system_t = α + β*institution_t + γ*system_{t-1} + ε
        4. CoVaR = α + β*VaR_institution + γ*median(system_{t-1})
        5. ΔCoVaR = CoVaR - VaR_system
        
        Parameters:
        -----------
        draws : int
            Number of posterior samples
        tune : int
            Number of tuning samples
        include_lagged : bool
            Whether to include lagged system returns as control
            
        Returns:
        --------
        results : dict
            Complete results including traces and estimates
        """
        print("=" * 80)
        print("CoVaR ESTIMATION USING PyMC (NUTS/HMC)")
        print("=" * 80)
        
        # Step 1: Estimate institution's VaR
        print(f"\nStep 1: Estimating institution VaR at α={self.alpha}...")
        institution_var, inst_trace, inst_var_samples = self.estimate_var_bayesian(
            self.institution_returns, self.alpha, draws=draws, tune=tune
        )
        print(f"  Institution VaR_{self.alpha}: {institution_var:.4f}")
        
        # Step 2: Estimate system's unconditional VaR
        print(f"\nStep 2: Estimating system VaR at α={self.alpha}...")
        system_var, sys_trace, sys_var_samples = self.estimate_var_bayesian(
            self.system_returns, self.alpha, draws=draws, tune=tune
        )
        print(f"  System VaR_{self.alpha}: {system_var:.4f}")
        
        # Step 3: Bayesian quantile regression for CoVaR
        print(f"\nStep 3: Bayesian quantile regression at α={self.alpha}...")
        
        # Prepare data
        if include_lagged:
            X_inst = self.institution_returns[1:]
            X_sys_lag = self.system_returns[:-1]
            y_sys = self.system_returns[1:]
        else:
            X_inst = self.institution_returns
            X_sys_lag = None
            y_sys = self.system_returns
        
        # Build quantile regression model
        with pm.Model() as covar_model:
            # Priors for regression coefficients
            alpha_coef = pm.Normal('alpha', mu=0, sigma=10)
            beta_coef = pm.Normal('beta', mu=0, sigma=5)
            
            if include_lagged:
                gamma_coef = pm.Normal('gamma', mu=0, sigma=5)
                mu = alpha_coef + beta_coef * X_inst + gamma_coef * X_sys_lag
            else:
                mu = alpha_coef + beta_coef * X_inst
            
            # Scale parameter
            b = pm.HalfNormal('b', sigma=1.0)
            
            # Likelihood
            y_obs = pm.CustomDist(
                'y_obs',
                mu, self.alpha, b,
                logp=asymmetric_laplace_logp,
                observed=y_sys
            )
            
            # Sample
            print(f"  Sampling quantile regression with NUTS...")
            qr_trace = pm.sample(
                draws=draws,
                tune=tune,
                cores=2,
                return_inferencedata=True,
                progressbar=True
            )
        
        # Extract coefficients
        alpha_samples = qr_trace.posterior['alpha'].values.flatten()
        beta_samples = qr_trace.posterior['beta'].values.flatten()
        
        alpha_hat = np.median(alpha_samples)
        beta_hat = np.median(beta_samples)
        
        if include_lagged:
            gamma_samples = qr_trace.posterior['gamma'].values.flatten()
            gamma_hat = np.median(gamma_samples)
            median_sys_lag = np.median(self.system_returns[:-1])
            print(f"\n  Quantile regression coefficients:")
            print(f"    α (intercept): {alpha_hat:.4f}")
            print(f"    β (institution): {beta_hat:.4f}")
            print(f"    γ (lagged system): {gamma_hat:.4f}")
        else:
            gamma_samples = None
            gamma_hat = 0
            median_sys_lag = 0
            print(f"\n  Quantile regression coefficients:")
            print(f"    α (intercept): {alpha_hat:.4f}")
            print(f"    β (institution): {beta_hat:.4f}")
        
        # Step 4: Calculate CoVaR and ΔCoVaR
        print(f"\nStep 4: Calculating CoVaR and ΔCoVaR...")
        
        # For each posterior sample, compute CoVaR
        if include_lagged:
            covar_samples = (alpha_samples + 
                           beta_samples * institution_var + 
                           gamma_samples * median_sys_lag)
        else:
            covar_samples = alpha_samples + beta_samples * institution_var
        
        # ΔCoVaR samples
        delta_covar_samples = covar_samples - sys_var_samples
        
        # Point estimates
        covar = np.median(covar_samples)
        delta_covar = np.median(delta_covar_samples)
        
        # Credible intervals
        covar_ci = np.percentile(covar_samples, [2.5, 97.5])
        delta_covar_ci = np.percentile(delta_covar_samples, [2.5, 97.5])
        
        print(f"\n  RESULTS:")
        print(f"    CoVaR_{self.alpha}: {covar:.4f}")
        print(f"    ΔCoVaR_{self.alpha}: {delta_covar:.4f}")
        print(f"\n  95% Credible Intervals:")
        print(f"    CoVaR: [{covar_ci[0]:.4f}, {covar_ci[1]:.4f}]")
        print(f"    ΔCoVaR: [{delta_covar_ci[0]:.4f}, {delta_covar_ci[1]:.4f}]")
        
        # Store results
        self.results = {
            'institution_var': institution_var,
            'institution_var_samples': inst_var_samples,
            'institution_trace': inst_trace,
            'system_var': system_var,
            'system_var_samples': sys_var_samples,
            'system_trace': sys_trace,
            'covar': covar,
            'covar_samples': covar_samples,
            'delta_covar': delta_covar,
            'delta_covar_samples': delta_covar_samples,
            'alpha_hat': alpha_hat,
            'beta_hat': beta_hat,
            'gamma_hat': gamma_hat if include_lagged else None,
            'alpha_samples': alpha_samples,
            'beta_samples': beta_samples,
            'gamma_samples': gamma_samples if include_lagged else None,
            'covar_ci': covar_ci,
            'delta_covar_ci': delta_covar_ci,
            'qr_trace': qr_trace
        }
        
        return self.results
    
    def print_diagnostics(self):
        """
        Print MCMC diagnostics for all models
        """
        print("\n" + "=" * 80)
        print("MCMC DIAGNOSTICS")
        print("=" * 80)
        
        # Institution VaR diagnostics
        print("\n1. Institution VaR Model:")
        print(az.summary(
            self.results['institution_trace'],
            var_names=['beta_0'],
            round_to=4
        ))
        
        # System VaR diagnostics
        print("\n2. System VaR Model:")
        print(az.summary(
            self.results['system_trace'],
            var_names=['beta_0'],
            round_to=4
        ))
        
        # Quantile regression diagnostics
        print("\n3. Quantile Regression Model:")
        var_names = ['alpha', 'beta']
        if self.results['gamma_hat'] is not None:
            var_names.append('gamma')
        print(az.summary(
            self.results['qr_trace'],
            var_names=var_names,
            round_to=4
        ))
        
        print("\nKey Diagnostics:")
        print("  - r_hat ≈ 1.0 indicates convergence")
        print("  - ess_bulk > 400 indicates sufficient effective sample size")
        print("  - ess_tail > 400 indicates good tail sampling")
    
    def plot_results(self, save_path=None):
        """
        Create comprehensive visualization of PyMC results
        """
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Institution returns with VaR
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.institution_returns, bins=50, alpha=0.7, density=True,
                color='steelblue', edgecolor='black')
        ax1.axvline(self.results['institution_var'], color='red', linestyle='--',
                   linewidth=2, label=f'VaR_{self.alpha}')
        ax1.set_xlabel('Institution Returns', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Institution Return Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: System returns with VaR and CoVaR
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.system_returns, bins=50, alpha=0.7, density=True,
                color='green', edgecolor='black')
        ax2.axvline(self.results['system_var'], color='blue', linestyle='--',
                   linewidth=2, label=f'System VaR_{self.alpha}')
        ax2.axvline(self.results['covar'], color='red', linestyle='--',
                   linewidth=2, label=f'CoVaR_{self.alpha}')
        ax2.set_xlabel('System Returns', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('System Return Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot with quantile regression
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(self.institution_returns, self.system_returns,
                   alpha=0.4, s=15, color='gray')
        x_range = np.linspace(self.institution_returns.min(),
                             self.institution_returns.max(), 100)
        y_quantile = self.results['alpha_hat'] + self.results['beta_hat'] * x_range
        ax3.plot(x_range, y_quantile, 'r-', linewidth=2,
                label=f'τ={self.alpha} quantile')
        ax3.plot(self.results['institution_var'], self.results['covar'],
                'ro', markersize=10, label='CoVaR point')
        ax3.set_xlabel('Institution Returns', fontsize=11)
        ax3.set_ylabel('System Returns', fontsize=11)
        ax3.set_title('Quantile Regression Fit', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Posterior of CoVaR
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(self.results['covar_samples'], bins=50, alpha=0.7,
                density=True, color='coral', edgecolor='black')
        ax4.axvline(self.results['covar'], color='red', linestyle='--',
                   linewidth=2, label='Median')
        ax4.axvline(self.results['covar_ci'][0], color='orange', linestyle=':',
                   linewidth=2, label='95% CI')
        ax4.axvline(self.results['covar_ci'][1], color='orange', linestyle=':',
                   linewidth=2)
        ax4.set_xlabel('CoVaR', fontsize=11)
        ax4.set_ylabel('Density', fontsize=11)
        ax4.set_title('Posterior Distribution of CoVaR', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Posterior of ΔCoVaR
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(self.results['delta_covar_samples'], bins=50, alpha=0.7,
                density=True, color='purple', edgecolor='black')
        ax5.axvline(self.results['delta_covar'], color='red', linestyle='--',
                   linewidth=2, label='Median')
        ax5.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax5.axvline(self.results['delta_covar_ci'][0], color='orange', linestyle=':',
                   linewidth=2, label='95% CI')
        ax5.axvline(self.results['delta_covar_ci'][1], color='orange', linestyle=':',
                   linewidth=2)
        ax5.set_xlabel('ΔCoVaR', fontsize=11)
        ax5.set_ylabel('Density', fontsize=11)
        ax5.set_title('Posterior Distribution of ΔCoVaR\n(Systemic Risk Contribution)',
                     fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Trace plot for beta coefficient
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(self.results['beta_samples'], linewidth=0.5, alpha=0.7, color='steelblue')
        ax6.axhline(self.results['beta_hat'], color='red', linestyle='--',
                   linewidth=2, label='Median')
        ax6.set_xlabel('Iteration', fontsize=11)
        ax6.set_ylabel('β (Institution coefficient)', fontsize=11)
        ax6.set_title('MCMC Trace: β Parameter', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Joint posterior of α and β
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter(self.results['alpha_samples'], self.results['beta_samples'],
                   alpha=0.3, s=5, color='navy')
        ax7.axvline(self.results['alpha_hat'], color='red', linestyle='--',
                   linewidth=1, alpha=0.7)
        ax7.axhline(self.results['beta_hat'], color='red', linestyle='--',
                   linewidth=1, alpha=0.7)
        ax7.set_xlabel('α (Intercept)', fontsize=11)
        ax7.set_ylabel('β (Institution)', fontsize=11)
        ax7.set_title('Joint Posterior: α and β', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Posterior comparison - using ArviZ forest plot style
        ax8 = fig.add_subplot(gs[2, 1])
        params = ['CoVaR', 'System VaR', 'ΔCoVaR']
        medians = [self.results['covar'], 
                   self.results['system_var'],
                   self.results['delta_covar']]
        ci_low = [self.results['covar_ci'][0],
                  np.percentile(self.results['system_var_samples'], 2.5),
                  self.results['delta_covar_ci'][0]]
        ci_high = [self.results['covar_ci'][1],
                   np.percentile(self.results['system_var_samples'], 97.5),
                   self.results['delta_covar_ci'][1]]
        
        y_pos = np.arange(len(params))
        ax8.errorbar(medians, y_pos, 
                     xerr=[np.array(medians) - np.array(ci_low),
                           np.array(ci_high) - np.array(medians)],
                     fmt='o', markersize=8, capsize=5, capthick=2,
                     color='darkblue', ecolor='steelblue', linewidth=2)
        ax8.set_yticks(y_pos)
        ax8.set_yticklabels(params)
        ax8.set_xlabel('Value', fontsize=11)
        ax8.set_title('Risk Measures Comparison\n(with 95% Credible Intervals)',
                     fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')
        ax8.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        # Plot 9: Autocorrelation of beta samples
        ax9 = fig.add_subplot(gs[2, 2])
        lags = np.arange(0, min(100, len(self.results['beta_samples'])//10))
        autocorr = [np.corrcoef(self.results['beta_samples'][:-lag if lag > 0 else None],
                               self.results['beta_samples'][lag:])[0,1] if lag > 0
                   else 1.0 for lag in lags]
        ax9.bar(lags, autocorr, color='steelblue', alpha=0.7, edgecolor='black')
        ax9.axhline(0, color='black', linestyle='-', linewidth=1)
        ax9.set_xlabel('Lag', fontsize=11)
        ax9.set_ylabel('Autocorrelation', fontsize=11)
        ax9.set_title('Autocorrelation of β Samples', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='y')
        
        # Plot 10-12: ArviZ-style diagnostic plots
        # ESS (Effective Sample Size)
        ax10 = fig.add_subplot(gs[3, 0])
        summary = az.summary(self.results['qr_trace'], var_names=['alpha', 'beta'])
        params_plot = summary.index.tolist()
        ess_bulk = summary['ess_bulk'].values
        ess_tail = summary['ess_tail'].values
        
        x = np.arange(len(params_plot))
        width = 0.35
        ax10.bar(x - width/2, ess_bulk, width, label='ESS Bulk', color='steelblue', alpha=0.7)
        ax10.bar(x + width/2, ess_tail, width, label='ESS Tail', color='coral', alpha=0.7)
        ax10.axhline(400, color='red', linestyle='--', linewidth=1, label='Threshold (400)')
        ax10.set_xticks(x)
        ax10.set_xticklabels(params_plot)
        ax10.set_ylabel('Effective Sample Size', fontsize=11)
        ax10.set_title('MCMC Effective Sample Size', fontsize=12, fontweight='bold')
        ax10.legend(fontsize=9)
        ax10.grid(True, alpha=0.3, axis='y')
        
        # R-hat convergence diagnostic
        ax11 = fig.add_subplot(gs[3, 1])
        r_hat = summary['r_hat'].values
        ax11.bar(params_plot, r_hat, color='green', alpha=0.7, edgecolor='black')
        ax11.axhline(1.0, color='blue', linestyle='--', linewidth=2, label='Perfect (1.0)')
        ax11.axhline(1.01, color='orange', linestyle='--', linewidth=1, label='Good (<1.01)')
        ax11.set_ylabel('R-hat', fontsize=11)
        ax11.set_title('Gelman-Rubin Convergence Diagnostic', fontsize=12, fontweight='bold')
        ax11.legend(fontsize=9)
        ax11.grid(True, alpha=0.3, axis='y')
        ax11.set_ylim([0.99, max(1.02, np.max(r_hat) * 1.01)])
        
        # Interpretation summary
        ax12 = fig.add_subplot(gs[3, 2])
        ax12.axis('off')
        
        interpretation = f"""
        INTERPRETATION SUMMARY
        ═══════════════════════
        
        Institution VaR: {self.results['institution_var']:.4f}
          → 5% probability of losses exceeding this
        
        System VaR: {self.results['system_var']:.4f}
          → Unconditional system risk
        
        CoVaR: {self.results['covar']:.4f}
          → System risk when institution in distress
        
        ΔCoVaR: {self.results['delta_covar']:.4f}
          → Systemic risk contribution
        """
        
        if self.results['delta_covar'] < -0.005:
            interpretation += "\n  ⚠ HIGHLY SYSTEMIC INSTITUTION"
        elif self.results['delta_covar'] < -0.001:
            interpretation += "\n  ⚠ Moderate systemic risk"
        else:
            interpretation += "\n  ✓ Limited systemic impact"
        
        interpretation += f"\n\nβ coefficient: {self.results['beta_hat']:.4f}"
        if abs(self.results['beta_hat']) > 0.5:
            interpretation += "\n  → Strong correlation with system"
        else:
            interpretation += "\n  → Weak correlation with system"
        
        ax12.text(0.05, 0.95, interpretation, transform=ax12.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('CoVaR Estimation with PyMC (NUTS/HMC)', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()


def generate_synthetic_data(n_obs=1000, correlation=0.6, seed=42):
    """
    Generate synthetic financial return data for demonstration
    """
    np.random.seed(seed)
    
    mean = [0, 0]
    cov = [[0.02**2, correlation * 0.02 * 0.015],
           [correlation * 0.02 * 0.015, 0.015**2]]
    
    returns = np.random.multivariate_normal(mean, cov, n_obs)
    
    # Add fat tails
    t_shocks = stats.t.rvs(df=5, size=(n_obs, 2)) * 0.5
    returns = 0.7 * returns + 0.3 * t_shocks
    
    system_returns = returns[:, 0]
    institution_returns = returns[:, 1]
    
    # Add mild autocorrelation
    for i in range(1, n_obs):
        system_returns[i] += 0.1 * system_returns[i-1]
    
    return system_returns, institution_returns


def main():
    """
    Main function demonstrating PyMC-based CoVaR estimation
    """
    print("=" * 80)
    print("SYSTEMIC RISK MEASUREMENT: CoVaR WITH PyMC")
    print("=" * 80)
    print("\nThis implementation uses PyMC with NUTS (No-U-Turn Sampler),")
    print("an advanced Hamiltonian Monte Carlo variant that:")
    print("  • Automatically tunes step size and trajectory length")
    print("  • Uses gradient information for efficient sampling")
    print("  • Provides built-in convergence diagnostics")
    print("  • Is much faster than basic Gibbs sampling")
    print("=" * 80)
    
    # Generate data
    print("\n\nGenerating synthetic financial data...")
    system_returns, institution_returns = generate_synthetic_data(
        n_obs=1000, correlation=0.6, seed=42
    )
    
    print(f"  Number of observations: {len(system_returns)}")
    print(f"  System returns - Mean: {np.mean(system_returns):.4f}, "
          f"Std: {np.std(system_returns):.4f}")
    print(f"  Institution returns - Mean: {np.mean(institution_returns):.4f}, "
          f"Std: {np.std(institution_returns):.4f}")
    print(f"  Correlation: {np.corrcoef(system_returns, institution_returns)[0,1]:.4f}")
    
    # Initialize estimator
    estimator = CoVaREstimatorPyMC(
        system_returns=system_returns,
        institution_returns=institution_returns,
        alpha=0.05
    )
    
    # Estimate CoVaR
    print("\n\nStarting CoVaR estimation with PyMC...")
    results = estimator.estimate_covar_bayesian(
        draws=2000,      # Posterior samples
        tune=1000,       # Warmup/tuning samples
        include_lagged=True
    )
    
    # Print diagnostics
    estimator.print_diagnostics()
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    print(f"\n1. Institution VaR: {results['institution_var']:.4f}")
    print("   → 5% probability of institution losses exceeding this level")
    
    print(f"\n2. System VaR (unconditional): {results['system_var']:.4f}")
    print("   → Under normal conditions, 5% chance system losses exceed this")
    
    print(f"\n3. CoVaR: {results['covar']:.4f}")
    print("   → System's 5% quantile WHEN institution is in distress")
    
    print(f"\n4. ΔCoVaR: {results['delta_covar']:.4f}")
    if results['delta_covar'] < -0.005:
        print("   → ⚠ HIGHLY SYSTEMIC INSTITUTION")
        print("   → System risk substantially increases when this institution fails")
    elif results['delta_covar'] < -0.001:
        print("   → Moderate systemic risk contributor")
    else:
        print("   → Limited systemic impact")
    
    print(f"\n5. β coefficient: {results['beta_hat']:.4f}")
    if abs(results['beta_hat']) > 0.5:
        print("   → Strong positive relationship with system")
    else:
        print("   → Moderate relationship")
    
    print("\n" + "=" * 80)
    
    # Visualize
    print("\nGenerating comprehensive visualizations...")
    estimator.plot_results(save_path='/mnt/user-data/outputs/covar_pymc_results.png')
    
    # Save results
    results_df = pd.DataFrame({
        'Metric': ['Institution VaR', 'System VaR', 'CoVaR', 'ΔCoVaR',
                   'α (Intercept)', 'β (Institution)', 'γ (Lagged)'],
        'Median': [
            results['institution_var'],
            results['system_var'],
            results['covar'],
            results['delta_covar'],
            results['alpha_hat'],
            results['beta_hat'],
            results['gamma_hat']
        ],
        'CI_Lower': [
            np.percentile(results['institution_var_samples'], 2.5),
            np.percentile(results['system_var_samples'], 2.5),
            results['covar_ci'][0],
            results['delta_covar_ci'][0],
            np.percentile(results['alpha_samples'], 2.5),
            np.percentile(results['beta_samples'], 2.5),
            np.percentile(results['gamma_samples'], 2.5) if results['gamma_samples'] is not None else np.nan
        ],
        'CI_Upper': [
            np.percentile(results['institution_var_samples'], 97.5),
            np.percentile(results['system_var_samples'], 97.5),
            results['covar_ci'][1],
            results['delta_covar_ci'][1],
            np.percentile(results['alpha_samples'], 97.5),
            np.percentile(results['beta_samples'], 97.5),
            np.percentile(results['gamma_samples'], 97.5) if results['gamma_samples'] is not None else np.nan
        ]
    })
    
    results_df.to_csv('/mnt/user-data/outputs/covar_pymc_results.csv', index=False)
    print("Results saved to /mnt/user-data/outputs/covar_pymc_results.csv")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Advantages of PyMC Implementation:")
    print("  ✓ Automatic tuning of MCMC parameters (no manual tuning needed)")
    print("  ✓ NUTS is much more efficient than basic Gibbs sampling")
    print("  ✓ Built-in diagnostics (R-hat, ESS) ensure reliability")
    print("  ✓ Cleaner, more readable probabilistic programming syntax")
    print("  ✓ Better handling of correlations via Hamiltonian dynamics")
    
    return results


if __name__ == "__main__":
    results = main()
