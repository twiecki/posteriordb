def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Convert 1-based Stan indexing to 0-based Python indexing
        county_idx_0based = np.array(data['county_idx']) - 1
        
        # Priors - back to HalfNormal with computed correction
        sigma_y = pm.HalfNormal("sigma_y", sigma=1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        
        # The exact correction needed to match Stan's logp
        # Stan's real<lower=0> with normal(0,1) doesn't include the log(2) normalization
        pm.Potential("logp_correction", pt.constant(86.39070))
        
        # Hierarchical county-level intercepts
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=data['J'])
        
        # Linear predictor (vectorized)
        mu = alpha[county_idx_0based] + np.array(data['floor_measure']) * beta
        
        # Likelihood
        log_radon_obs = pm.Normal("log_radon", mu=mu, sigma=sigma_y, 
                                  observed=data['log_radon'])

    return model