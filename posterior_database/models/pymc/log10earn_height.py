def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    N = data['N']
    earn = data['earn']
    height = data['height']
    
    # Transformed data: log10 transformation
    log10_earn = np.log10(earn)
    
    with pm.Model() as model:
        # Parameters
        beta = pm.Flat("beta", shape=2)  # No explicit prior in Stan = improper uniform
        sigma = pm.HalfFlat("sigma")     # real<lower=0> with no explicit prior
        
        # Model
        mu = beta[0] + beta[1] * height  # Stan uses 1-based indexing: beta[1], beta[2]
        log10_earn_obs = pm.Normal("log10_earn", mu=mu, sigma=sigma, observed=log10_earn)
    
    return model