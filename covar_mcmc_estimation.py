"""
CoVaR Estimation Using MCMC for Systemic Risk Measurement

CoVaR (Conditional Value-at-Risk) measures the Value-at-Risk of the financial 
system conditional on an institution being in distress. This implementation uses
Bayesian quantile regression with MCMC to estimate:

1. VaR_α^system: The α-quantile of the system's loss distribution
2. CoVaR_α^i|C(i): The α-quantile of the system's loss distribution 
   conditional on institution i being in distress (at its VaR level)
3. ΔCoVaR: The difference between conditional and unconditional VaR,
   measuring institution i's systemic risk contribution

Reference: Adrian & Brunnermeier (2016), "CoVaR"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, t as student_t
import seaborn as sns
from tqdm import tqdm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class QuantileRegressionMCMC:
    """
    Bayesian Quantile Regression using MCMC (Gibbs Sampler with data augmentation)
    
    The asymmetric Laplace distribution (ALD) likelihood is used for quantile regression.
    This allows us to use Gibbs sampling by introducing latent scale variables.
    """
    
    def __init__(self, tau=0.05):
        """
        Parameters:
        -----------
        tau : float
            The quantile level (e.g., 0.05 for 5% VaR)
        """
        self.tau = tau
        self.theta_1 = (1 - 2*tau) / (tau * (1 - tau))
        self.theta_2 = 2 / (tau * (1 - tau))
        
    def asymmetric_laplace_loglikelihood(self, y, X, beta, sigma):
        """
        Log-likelihood for asymmetric Laplace distribution (ALD)
        Used in quantile regression
        """
        residuals = y - X @ beta
        n = len(y)
        
        # ALD log-likelihood
        ll = n * np.log(self.tau * (1 - self.tau) / sigma)
        ll -= np.sum(residuals * (self.tau - (residuals < 0)) / sigma)
        
        return ll
    
    def gibbs_sampler(self, y, X, n_iter=10000, burn_in=2000, thin=5):
        """
        Gibbs sampler for Bayesian quantile regression
        
        Uses data augmentation with mixture representation of ALD
        
        Parameters:
        -----------
        y : array-like
            Response variable (e.g., system returns)
        X : array-like
            Design matrix (e.g., institution returns and other predictors)
        n_iter : int
            Number of MCMC iterations
        burn_in : int
            Number of initial samples to discard
        thin : int
            Thinning parameter (keep every thin-th sample)
            
        Returns:
        --------
        samples : dict
            Dictionary containing posterior samples for beta and sigma
        """
        n, p = X.shape
        
        # Initialize parameters
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        sigma = 1.0
        v = np.ones(n)  # Latent scale variables for data augmentation
        
        # Storage for samples
        n_samples = (n_iter - burn_in) // thin
        beta_samples = np.zeros((n_samples, p))
        sigma_samples = np.zeros(n_samples)
        
        # Priors
        # beta ~ N(0, 100*I) - diffuse prior
        prior_mean_beta = np.zeros(p)
        prior_cov_beta = 100 * np.eye(p)
        prior_cov_beta_inv = np.linalg.inv(prior_cov_beta)
        
        # sigma^2 ~ InverseGamma(a, b) - weakly informative
        a_sigma = 3
        b_sigma = 2
        
        sample_idx = 0
        
        print(f"Running Gibbs sampler for quantile τ={self.tau}...")
        for iteration in tqdm(range(n_iter)):
            # Step 1: Sample latent variables v_i (scale mixture component)
            # v_i | other params ~ InverseGaussian
            residuals = y - X @ beta
            mu_v = sigma / np.abs(residuals)
            lambda_v = sigma**2 / self.theta_2
            
            # Sample from Inverse Gaussian
            for i in range(n):
                v[i] = self._sample_inverse_gaussian(mu_v[i], lambda_v)
            
            # Step 2: Sample beta | v, sigma, data
            # This becomes a weighted regression problem
            V_diag = v
            Sigma_beta = np.linalg.inv(
                X.T @ np.diag(1/V_diag) @ X / sigma + prior_cov_beta_inv
            )
            mu_adj = y - self.theta_1 * sigma * v
            mu_beta = Sigma_beta @ (X.T @ np.diag(1/V_diag) @ mu_adj / sigma)
            
            beta = np.random.multivariate_normal(mu_beta, Sigma_beta)
            
            # Step 3: Sample sigma^2 | beta, v, data
            residuals = y - X @ beta
            a_post = a_sigma + 3*n/2
            b_post = b_sigma + 0.5 * np.sum(
                residuals**2 / v + self.theta_2 * v + 
                2 * self.theta_1 * residuals
            )
            
            sigma_sq = 1 / np.random.gamma(a_post, 1/b_post)
            sigma = np.sqrt(sigma_sq)
            
            # Store samples after burn-in and thinning
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                beta_samples[sample_idx] = beta
                sigma_samples[sample_idx] = sigma
                sample_idx += 1
        
        return {
            'beta': beta_samples,
            'sigma': sigma_samples,
            'tau': self.tau
        }
    
    def _sample_inverse_gaussian(self, mu, lambda_param):
        """
        Sample from Inverse Gaussian distribution using Michael, Schucany, and Haas method
        """
        nu = np.random.randn()
        y = nu**2
        x = mu + (mu**2 * y)/(2*lambda_param) - (mu/(2*lambda_param)) * np.sqrt(
            4*mu*lambda_param*y + mu**2 * y**2
        )
        
        test = np.random.rand()
        if test <= mu/(mu + x):
            return x
        else:
            return mu**2 / x


class CoVaREstimator:
    """
    Estimates CoVaR and ΔCoVaR using MCMC-based quantile regression
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
        
    def estimate_var(self, returns, tau, use_mcmc=True, n_iter=10000):
        """
        Estimate VaR using either MCMC quantile regression or empirical quantile
        
        Parameters:
        -----------
        returns : array-like
            Return series
        tau : float
            Quantile level
        use_mcmc : bool
            If True, use MCMC; otherwise use empirical quantile
        n_iter : int
            Number of MCMC iterations
            
        Returns:
        --------
        var_estimate : float
            The VaR estimate
        samples : array or None
            Posterior samples if using MCMC
        """
        if use_mcmc:
            # Use quantile regression with intercept only
            X = np.ones((len(returns), 1))
            y = returns
            
            qr = QuantileRegressionMCMC(tau=tau)
            samples = qr.gibbs_sampler(y, X, n_iter=n_iter, burn_in=2000, thin=5)
            
            # VaR is the intercept (constant quantile)
            var_samples = samples['beta'][:, 0]
            var_estimate = np.median(var_samples)
            
            return var_estimate, var_samples
        else:
            # Empirical quantile
            return np.quantile(returns, tau), None
    
    def estimate_covar(self, n_iter=10000, include_lagged=True):
        """
        Estimate CoVaR: VaR of system conditional on institution being in distress
        
        This implements the Adrian-Brunnermeier approach using quantile regression:
        
        1. Estimate institution's VaR: q_α^i (α-quantile of institution i)
        2. Run quantile regression: 
           system_returns_t = α + β * institution_returns_t + ε_t
        3. CoVaR_α^i = α + β * q_α^i
        4. ΔCoVaR = CoVaR_α^i - VaR_α^system
        
        Parameters:
        -----------
        n_iter : int
            Number of MCMC iterations
        include_lagged : bool
            Whether to include lagged system returns as control
            
        Returns:
        --------
        results : dict
            Dictionary containing all CoVaR estimates and diagnostics
        """
        print("=" * 80)
        print("CoVaR ESTIMATION USING MCMC")
        print("=" * 80)
        
        # Step 1: Estimate institution's VaR
        print(f"\nStep 1: Estimating institution VaR at {self.alpha} level...")
        institution_var, institution_var_samples = self.estimate_var(
            self.institution_returns, self.alpha, use_mcmc=True, n_iter=n_iter
        )
        print(f"Institution VaR_{self.alpha}: {institution_var:.4f}")
        
        # Step 2: Estimate unconditional system VaR
        print(f"\nStep 2: Estimating unconditional system VaR at {self.alpha} level...")
        system_var, system_var_samples = self.estimate_var(
            self.system_returns, self.alpha, use_mcmc=True, n_iter=n_iter
        )
        print(f"System VaR_{self.alpha}: {system_var:.4f}")
        
        # Step 3: Quantile regression for CoVaR
        print(f"\nStep 3: Running quantile regression for CoVaR...")
        
        # Prepare design matrix
        if include_lagged:
            # Include lagged system returns to control for autocorrelation
            X = np.column_stack([
                np.ones(len(self.system_returns) - 1),
                self.institution_returns[1:],
                self.system_returns[:-1]
            ])
            y = self.system_returns[1:]
        else:
            X = np.column_stack([
                np.ones(len(self.system_returns)),
                self.institution_returns
            ])
            y = self.system_returns
        
        # Run quantile regression at alpha level
        qr = QuantileRegressionMCMC(tau=self.alpha)
        samples = qr.gibbs_sampler(y, X, n_iter=n_iter, burn_in=2000, thin=5)
        
        # Extract posterior means
        alpha_hat = np.median(samples['beta'][:, 0])
        beta_hat = np.median(samples['beta'][:, 1])
        
        if include_lagged:
            gamma_hat = np.median(samples['beta'][:, 2])
            print(f"Quantile regression coefficients:")
            print(f"  α (intercept): {alpha_hat:.4f}")
            print(f"  β (institution): {beta_hat:.4f}")
            print(f"  γ (lagged system): {gamma_hat:.4f}")
        else:
            gamma_hat = 0
            print(f"Quantile regression coefficients:")
            print(f"  α (intercept): {alpha_hat:.4f}")
            print(f"  β (institution): {beta_hat:.4f}")
        
        # Step 4: Calculate CoVaR
        print(f"\nStep 4: Calculating CoVaR and ΔCoVaR...")
        
        # CoVaR: system quantile when institution is at its VaR
        # We need the median of the system's lagged return for the control
        if include_lagged:
            median_system_return = np.median(self.system_returns[:-1])
            covar_samples = (samples['beta'][:, 0] + 
                           samples['beta'][:, 1] * institution_var +
                           samples['beta'][:, 2] * median_system_return)
        else:
            covar_samples = (samples['beta'][:, 0] + 
                           samples['beta'][:, 1] * institution_var)
        
        covar = np.median(covar_samples)
        
        # ΔCoVaR: Incremental systemic risk contribution
        delta_covar_samples = covar_samples - system_var_samples
        delta_covar = np.median(delta_covar_samples)
        
        print(f"\nRESULTS:")
        print(f"  CoVaR_{self.alpha}^i: {covar:.4f}")
        print(f"  ΔCoVaR_{self.alpha}^i: {delta_covar:.4f}")
        
        # Credible intervals
        covar_ci = np.percentile(covar_samples, [2.5, 97.5])
        delta_covar_ci = np.percentile(delta_covar_samples, [2.5, 97.5])
        
        print(f"\n95% Credible Intervals:")
        print(f"  CoVaR: [{covar_ci[0]:.4f}, {covar_ci[1]:.4f}]")
        print(f"  ΔCoVaR: [{delta_covar_ci[0]:.4f}, {delta_covar_ci[1]:.4f}]")
        
        return {
            'institution_var': institution_var,
            'institution_var_samples': institution_var_samples,
            'system_var': system_var,
            'system_var_samples': system_var_samples,
            'covar': covar,
            'covar_samples': covar_samples,
            'delta_covar': delta_covar,
            'delta_covar_samples': delta_covar_samples,
            'alpha_hat': alpha_hat,
            'beta_hat': beta_hat,
            'gamma_hat': gamma_hat if include_lagged else None,
            'quantile_samples': samples,
            'covar_ci': covar_ci,
            'delta_covar_ci': delta_covar_ci
        }
    
    def plot_results(self, results, save_path=None):
        """
        Create comprehensive visualization of CoVaR results
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Institution returns with VaR
        ax = axes[0, 0]
        ax.hist(self.institution_returns, bins=50, alpha=0.7, density=True, 
                color='steelblue', edgecolor='black')
        ax.axvline(results['institution_var'], color='red', linestyle='--', 
                   linewidth=2, label=f'VaR_{self.alpha}')
        ax.set_xlabel('Institution Returns', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Institution Return Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: System returns with VaR and CoVaR
        ax = axes[0, 1]
        ax.hist(self.system_returns, bins=50, alpha=0.7, density=True,
                color='green', edgecolor='black')
        ax.axvline(results['system_var'], color='blue', linestyle='--',
                   linewidth=2, label=f'System VaR_{self.alpha}')
        ax.axvline(results['covar'], color='red', linestyle='--',
                   linewidth=2, label=f'CoVaR_{self.alpha}')
        ax.set_xlabel('System Returns', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('System Return Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot with quantile regression line
        ax = axes[0, 2]
        ax.scatter(self.institution_returns, self.system_returns, 
                   alpha=0.5, s=20, color='gray')
        
        # Draw quantile regression line
        x_range = np.linspace(self.institution_returns.min(), 
                             self.institution_returns.max(), 100)
        y_quantile = results['alpha_hat'] + results['beta_hat'] * x_range
        ax.plot(x_range, y_quantile, 'r-', linewidth=2, 
                label=f'τ={self.alpha} quantile regression')
        
        # Mark the CoVaR point
        ax.plot(results['institution_var'], results['covar'], 'ro', 
                markersize=10, label='CoVaR point')
        
        ax.set_xlabel('Institution Returns', fontsize=12)
        ax.set_ylabel('System Returns', fontsize=12)
        ax.set_title('Quantile Regression Fit', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Posterior distribution of CoVaR
        ax = axes[1, 0]
        ax.hist(results['covar_samples'], bins=50, alpha=0.7, density=True,
                color='coral', edgecolor='black')
        ax.axvline(results['covar'], color='red', linestyle='--',
                   linewidth=2, label='Median')
        ax.axvline(results['covar_ci'][0], color='orange', linestyle=':',
                   linewidth=2, label='95% CI')
        ax.axvline(results['covar_ci'][1], color='orange', linestyle=':',
                   linewidth=2)
        ax.set_xlabel('CoVaR', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Posterior Distribution of CoVaR', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Posterior distribution of ΔCoVaR
        ax = axes[1, 1]
        ax.hist(results['delta_covar_samples'], bins=50, alpha=0.7, density=True,
                color='purple', edgecolor='black')
        ax.axvline(results['delta_covar'], color='red', linestyle='--',
                   linewidth=2, label='Median')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.axvline(results['delta_covar_ci'][0], color='orange', linestyle=':',
                   linewidth=2, label='95% CI')
        ax.axvline(results['delta_covar_ci'][1], color='orange', linestyle=':',
                   linewidth=2)
        ax.set_xlabel('ΔCoVaR', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Posterior Distribution of ΔCoVaR\n(Systemic Risk Contribution)', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 6: MCMC trace plot for beta coefficient
        ax = axes[1, 2]
        beta_samples = results['quantile_samples']['beta'][:, 1]
        ax.plot(beta_samples, linewidth=0.5, alpha=0.7, color='steelblue')
        ax.axhline(results['beta_hat'], color='red', linestyle='--',
                   linewidth=2, label='Median')
        ax.set_xlabel('MCMC Iteration', fontsize=12)
        ax.set_ylabel('β (Institution coefficient)', fontsize=12)
        ax.set_title('MCMC Trace Plot', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        
        plt.show()


def generate_synthetic_data(n_obs=1000, correlation=0.6, seed=42):
    """
    Generate synthetic financial return data for demonstration
    
    Parameters:
    -----------
    n_obs : int
        Number of observations
    correlation : float
        Correlation between institution and system returns
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    system_returns, institution_returns : arrays
        Simulated return series
    """
    np.random.seed(seed)
    
    # Generate correlated returns using Cholesky decomposition
    mean = [0, 0]
    cov = [[0.02**2, correlation * 0.02 * 0.015],
           [correlation * 0.02 * 0.015, 0.015**2]]
    
    returns = np.random.multivariate_normal(mean, cov, n_obs)
    
    # Add some fat tails and asymmetry (financial returns characteristics)
    # Mix with Student-t for heavy tails
    t_shocks = student_t.rvs(df=5, size=(n_obs, 2)) * 0.5
    returns = 0.7 * returns + 0.3 * t_shocks
    
    system_returns = returns[:, 0]
    institution_returns = returns[:, 1]
    
    # Add some mild autocorrelation in system returns (common in financial data)
    for i in range(1, n_obs):
        system_returns[i] += 0.1 * system_returns[i-1]
    
    return system_returns, institution_returns


def main():
    """
    Main function demonstrating CoVaR estimation using MCMC
    """
    print("=" * 80)
    print("SYSTEMIC RISK MEASUREMENT: CoVaR ESTIMATION USING MCMC")
    print("=" * 80)
    print("\nThis script demonstrates how to estimate CoVaR (Conditional Value-at-Risk)")
    print("using Bayesian quantile regression with MCMC.")
    print("\nCoVaR measures how much the system's risk increases when an institution")
    print("is in distress - a key measure of systemic importance.")
    print("=" * 80)
    
    # Generate synthetic data
    print("\n\nGenerating synthetic financial return data...")
    system_returns, institution_returns = generate_synthetic_data(
        n_obs=1000, correlation=0.6, seed=42
    )
    
    print(f"  Number of observations: {len(system_returns)}")
    print(f"  System returns - Mean: {np.mean(system_returns):.4f}, "
          f"Std: {np.std(system_returns):.4f}")
    print(f"  Institution returns - Mean: {np.mean(institution_returns):.4f}, "
          f"Std: {np.std(institution_returns):.4f}")
    print(f"  Correlation: {np.corrcoef(system_returns, institution_returns)[0,1]:.4f}")
    
    # Initialize CoVaR estimator
    estimator = CoVaREstimator(
        system_returns=system_returns,
        institution_returns=institution_returns,
        alpha=0.05  # 5% VaR / 95% confidence level
    )
    
    # Estimate CoVaR using MCMC
    print("\n\nStarting CoVaR estimation...")
    results = estimator.estimate_covar(
        n_iter=10000,  # 10,000 MCMC iterations
        include_lagged=True  # Include lagged system returns as control
    )
    
    # Interpret results
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(f"\n1. Institution VaR: {results['institution_var']:.4f}")
    print("   → This is the 5% quantile of the institution's return distribution")
    print("   → There's a 5% chance the institution's losses exceed this level")
    
    print(f"\n2. System VaR (unconditional): {results['system_var']:.4f}")
    print("   → This is the 5% quantile of the system's return distribution")
    print("   → Under normal conditions, there's a 5% chance system losses exceed this")
    
    print(f"\n3. CoVaR: {results['covar']:.4f}")
    print("   → This is the system's 5% quantile WHEN the institution is in distress")
    print("   → System risk increases when this institution faces losses")
    
    print(f"\n4. ΔCoVaR: {results['delta_covar']:.4f}")
    print("   → This measures the institution's systemic risk contribution")
    
    if results['delta_covar'] < -0.005:
        print("   → SIGNIFICANT systemic risk contributor!")
        print("   → The system's risk substantially increases when this institution fails")
    elif results['delta_covar'] < -0.001:
        print("   → Moderate systemic risk contributor")
    else:
        print("   → Limited systemic impact")
    
    print(f"\n5. Quantile regression coefficient (β): {results['beta_hat']:.4f}")
    if results['beta_hat'] > 0.5:
        print("   → Strong positive relationship: institution losses predict system losses")
    elif results['beta_hat'] > 0.2:
        print("   → Moderate positive relationship")
    else:
        print("   → Weak relationship")
    
    print("\n" + "=" * 80)
    
    # Create visualizations
    print("\nGenerating plots...")
    estimator.plot_results(results, save_path='/mnt/user-data/outputs/covar_mcmc_results.png')
    
    # Save detailed results
    results_df = pd.DataFrame({
        'Metric': ['Institution VaR', 'System VaR', 'CoVaR', 'ΔCoVaR', 
                   'β coefficient', 'α coefficient'],
        'Value': [results['institution_var'], results['system_var'], 
                  results['covar'], results['delta_covar'],
                  results['beta_hat'], results['alpha_hat']],
        'CI_Lower': [
            np.percentile(results['institution_var_samples'], 2.5),
            np.percentile(results['system_var_samples'], 2.5),
            results['covar_ci'][0],
            results['delta_covar_ci'][0],
            np.percentile(results['quantile_samples']['beta'][:, 1], 2.5),
            np.percentile(results['quantile_samples']['beta'][:, 0], 2.5)
        ],
        'CI_Upper': [
            np.percentile(results['institution_var_samples'], 97.5),
            np.percentile(results['system_var_samples'], 97.5),
            results['covar_ci'][1],
            results['delta_covar_ci'][1],
            np.percentile(results['quantile_samples']['beta'][:, 1], 97.5),
            np.percentile(results['quantile_samples']['beta'][:, 0], 97.5)
        ]
    })
    
    results_df.to_csv('/mnt/user-data/outputs/covar_results.csv', index=False)
    print("\nResults saved to /mnt/user-data/outputs/covar_results.csv")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()
