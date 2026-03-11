def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    N = data['N']
    earn = data['earn']
    height = data['height']
    
    # Transformed data: log transformation (matching Stan's transformed data block)
    log_earn = np.log(earn)
    
    with pm.Model() as model:
        # Parameters
        # beta is a vector[2] in Stan - no explicit prior means improper uniform
        beta = pm.Flat("beta", shape=2)
        
        # sigma is real<lower=0> - no explicit prior means improper uniform on (0, inf)
        sigma = pm.HalfFlat("sigma")
        
        # Model: log_earn ~ normal(beta[1] + beta[2] * height, sigma)
        # Note: Stan uses 1-based indexing, so beta[1] is beta[0] and beta[2] is beta[1] in Python
        mu = beta[0] + beta[1] * height
        log_earn_obs = pm.Normal("log_earn", mu=mu, sigma=sigma, observed=log_earn)
    
    return model