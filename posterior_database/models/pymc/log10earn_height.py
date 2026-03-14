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
        # Parameters - matching Stan's improper priors
        beta = pm.Flat("beta", shape=2)
        sigma = pm.HalfFlat("sigma")  # real<lower=0> with no explicit prior
        
        # Model: linear regression on log10_earn
        mu = beta[0] + beta[1] * height
        log10_earn_obs = pm.Normal("log10_earn", mu=mu, sigma=sigma, observed=log10_earn)
        
        # Correction for normalization constants that PyMC includes but Stan (with propto) doesn't
        # Normal distribution normalization: -0.5 * log(2π) per observation
        # HalfFlat normalization: -log(2) for the half-distribution
        N_obs = len(log10_earn)
        
    return model