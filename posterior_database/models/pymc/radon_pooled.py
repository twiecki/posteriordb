def make_model(data: dict):
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    N = data['N']
    floor_measure = data['floor_measure']
    log_radon = data['log_radon']
    
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        # sigma_y has normal(0, 1) prior with lower=0 constraint
        sigma_y = pm.HalfNormal("sigma_y", sigma=1)
        
        # Linear model
        mu = alpha + beta * floor_measure
        
        # Likelihood
        log_radon_obs = pm.Normal("log_radon", mu=mu, sigma=sigma_y, observed=log_radon)
        
    return model