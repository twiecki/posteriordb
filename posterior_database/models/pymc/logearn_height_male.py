def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Transformed data: log transformation (same as Stan's transformed data block)
    log_earn = np.log(data['earn'])
    
    with pm.Model() as model:
        # Parameters
        beta = pm.Flat("beta", shape=3)  # No explicit prior in Stan = improper uniform
        sigma = pm.HalfFlat("sigma")     # real<lower=0> with no explicit prior
        
        # Linear predictor: beta[1] + beta[2] * height + beta[3] * male
        # Stan indexing is 1-based, so beta[1] is beta[0] in Python, etc.
        mu = beta[0] + beta[1] * data['height'] + beta[2] * data['male']
        
        # Likelihood
        log_earn_obs = pm.Normal("log_earn", mu=mu, sigma=sigma, observed=log_earn)
    
    return model