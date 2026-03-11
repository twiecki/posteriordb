def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        N = data['N']
        J = data['J']
        county_idx = np.array(data['county_idx']) - 1  # Convert from 1-based to 0-based indexing
        floor_measure = np.array(data['floor_measure'])
        log_radon = np.array(data['log_radon'])
        
        # Parameters with priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=10)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1)
        
        # Hierarchical coefficients
        beta = pm.Normal("beta", mu=mu_beta, sigma=sigma_beta, shape=J)
        
        # Linear predictor
        mu = alpha + floor_measure * beta[county_idx]
        
        # Likelihood
        y_obs = pm.Normal("log_radon", mu=mu, sigma=sigma_y, observed=log_radon)

    return model