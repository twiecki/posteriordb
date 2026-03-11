def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    N = data['N']
    J = data['J'] 
    county_idx = np.array(data['county_idx']) - 1  # Convert from 1-based to 0-based indexing
    log_radon = data['log_radon']

    with pm.Model() as model:
        # Priors
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=J)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1)
        
        # Transformed parameters (non-centered parameterization)
        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_raw)
        
        # Likelihood
        mu = alpha[county_idx]
        y_obs = pm.Normal("log_radon", mu=mu, sigma=sigma_y, observed=log_radon)

    return model