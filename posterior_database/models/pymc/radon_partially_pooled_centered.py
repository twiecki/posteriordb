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
        sigma_y = pm.HalfNormal("sigma_y", sigma=1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1) 
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        
        # County-level random effects
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=J)
        
        # Likelihood
        mu = alpha[county_idx]
        pm.Normal("log_radon", mu=mu, sigma=sigma_y, observed=log_radon)
        
        # Correction for half-normal distributions (Stan vs PyMC logp difference)
        pm.Potential("half_dist_correction", -2 * pt.log(2.0))
        
    return model