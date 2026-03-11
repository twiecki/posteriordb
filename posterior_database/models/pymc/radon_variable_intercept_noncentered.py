def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        N = data['N']
        J = data['J']
        county_idx = np.array(data['county_idx']) - 1  # Convert 1-based to 0-based indexing
        floor_measure = np.array(data['floor_measure'])
        log_radon = np.array(data['log_radon'])
        
        # Parameters
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=J)
        beta = pm.Normal("beta", mu=0, sigma=10)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1)
        
        # Transformed parameters
        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_raw)
        
        # Model (vectorized)
        mu = alpha[county_idx] + floor_measure * beta
        y_obs = pm.Normal("log_radon", mu=mu, sigma=sigma_y, observed=log_radon)

    return model