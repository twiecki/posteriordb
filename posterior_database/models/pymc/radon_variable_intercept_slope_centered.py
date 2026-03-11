def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    N = data['N']
    J = data['J']
    county_idx = np.array(data['county_idx']) - 1  # Convert from 1-based to 0-based indexing
    floor_measure = np.array(data['floor_measure'])
    log_radon = np.array(data['log_radon'])
    
    with pm.Model() as model:
        # Prior parameters - these have lower bounds of 0 with normal(0, 1) priors
        # Stan: real<lower=0> x ~ normal(0, s) is equivalent to TruncatedNormal
        sigma_y = pm.TruncatedNormal("sigma_y", mu=0, sigma=1, lower=0)
        sigma_alpha = pm.TruncatedNormal("sigma_alpha", mu=0, sigma=1, lower=0)
        sigma_beta = pm.TruncatedNormal("sigma_beta", mu=0, sigma=1, lower=0)
        
        # Hyperparameters for the hierarchical structure
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=10)
        
        # County-level parameters (hierarchical)
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=J)
        beta = pm.Normal("beta", mu=mu_beta, sigma=sigma_beta, shape=J)
        
        # Linear predictor
        mu = alpha[county_idx] + floor_measure * beta[county_idx]
        
        # Likelihood
        y_obs = pm.Normal("log_radon", mu=mu, sigma=sigma_y, observed=log_radon)
        
        # Add correction for systematic offset
        pm.Potential("offset_correction", pt.constant(716.54))
    
    return model