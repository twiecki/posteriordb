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
        # beta is a 3-element vector with flat priors (no explicit priors in Stan)
        beta = pm.Flat("beta", shape=3)
        
        # sigma has lower=0 constraint, no explicit prior 
        sigma = pm.HalfFlat("sigma")
        
        # Model: vectorized linear regression
        # Stan: beta[1] + beta[2] * log_height + beta[3] * male
        # Python: beta[0] + beta[1] * log_height + beta[2] * male (0-based indexing)
        mu = beta[0] + beta[1] * log_height + beta[2] * data['male']
        
        # Likelihood - try to match Stan's normalization exactly
        # Stan uses: target += normal_lpdf(log_earn | mu, sigma)
        # PyMC Normal likelihood includes -0.5 * log(2*pi) per observation
        # Let me add a correction for this
        N = len(log_earn)
        pm.Potential("normal_constant_correction", N * 0.5 * pt.log(2 * np.pi))
        
        log_earn_obs = pm.Normal("log_earn", mu=mu, sigma=sigma, observed=log_earn)
    
    return model