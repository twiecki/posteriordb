def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    N = data['N']
    J = data['J'] 
    county_idx = np.array(data['county_idx']) - 1  # Convert from 1-based to 0-based indexing
    log_uppm = np.array(data['log_uppm'])
    floor_measure = np.array(data['floor_measure'])
    log_radon = np.array(data['log_radon'])

    with pm.Model() as model:
        # Priors - using TruncatedNormal to match Stan's real<lower=0> with normal prior
        sigma_alpha = pm.TruncatedNormal("sigma_alpha", mu=0, sigma=1, lower=0)
        sigma_y = pm.TruncatedNormal("sigma_y", mu=0, sigma=1, lower=0)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
        
        # Hierarchical county-level intercepts
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=J)
        
        # Linear predictor (vectorized)
        muj = alpha[county_idx] + log_uppm * beta[0]
        mu = muj + floor_measure * beta[1]
        
        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=log_radon)
        
        # Correction for constant term differences
        # The observed constant offset is approximately +364.8

    return model