def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Convert 1-based county indices to 0-based
        county_idx = np.array(data['county_idx']) - 1
        
        # Priors
        sigma_y = pm.HalfNormal("sigma_y", sigma=1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1) 
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        
        # Hierarchical county effects
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=data['J'])
        
        # Linear predictor
        mu = alpha[county_idx] + np.array(data['floor_measure']) * beta
        
        # Likelihood
        log_radon_obs = pm.Normal("log_radon", mu=mu, sigma=sigma_y, observed=data['log_radon'])

    return model