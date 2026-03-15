def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        N = data['N']
        earn = data['earn']
        height = data['height']
        
        # Parameters
        # beta is a vector of length 2 - using Flat priors since Stan doesn't specify priors
        beta = pm.Flat("beta", shape=2)
        
        # sigma is constrained to be positive - using HalfFlat since no prior specified
        sigma = pm.HalfFlat("sigma")
        
        # Model: earn ~ normal(beta[1] + beta[2] * height, sigma)
        # Note: Stan uses 1-based indexing, so beta[1] is beta[0] and beta[2] is beta[1] in Python
        mu = beta[0] + beta[1] * height
        
        # Likelihood
        earn_obs = pm.Normal("earn", mu=mu, sigma=sigma, observed=earn)

    return model