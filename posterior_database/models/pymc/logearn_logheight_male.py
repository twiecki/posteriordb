def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Transformed data - compute log transformations
    log_earn = np.log(data['earn'])
    log_height = np.log(data['height'])
    
    with pm.Model() as model:
        # Parameters
        # beta is a vector of length 3 with no explicit prior (improper uniform)
        beta = pm.Flat("beta", shape=3)
        
        # sigma is positive with no explicit prior
        sigma = pm.HalfFlat("sigma")
        
        # Model: vectorized linear regression
        # log_earn ~ normal(beta[0] + beta[1] * log_height + beta[2] * male, sigma)
        mu = beta[0] + beta[1] * log_height + beta[2] * data['male']
        
        # Likelihood
        log_earn_obs = pm.Normal("log_earn", mu=mu, sigma=sigma, observed=log_earn)
    
    return model