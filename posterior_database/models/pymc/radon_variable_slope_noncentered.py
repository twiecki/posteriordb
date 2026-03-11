def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data and ensure numpy arrays
        N = data['N']
        J = data['J'] 
        county_idx = np.array(data['county_idx']) - 1  # Convert 1-based to 0-based indexing
        floor_measure = np.array(data['floor_measure'])
        log_radon = np.array(data['log_radon'])
        
        # Parameters
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        
        # Non-centered parameterization for hierarchical effects
        beta_raw = pm.Normal("beta_raw", mu=0, sigma=1, shape=J)
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=10)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1)
        
        # Transformed parameters
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * beta_raw)
        
        # Likelihood
        mu = alpha + floor_measure * beta[county_idx]
        log_radon_obs = pm.Normal("log_radon", mu=mu, sigma=sigma_y, observed=log_radon)

    return model