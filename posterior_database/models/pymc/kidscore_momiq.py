def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Parameters
        # beta is a vector of length 2 with no explicit prior in Stan (improper uniform)
        beta = pm.Flat("beta", shape=2)
        
        # sigma has Cauchy(0, 2.5) prior with lower bound of 0
        sigma = pm.HalfCauchy("sigma", beta=2.5)
        
        # Model: kid_score ~ normal(beta[1] + beta[2] * mom_iq, sigma)
        # Note: Stan uses 1-based indexing, so beta[1] is beta[0] in Python
        mu = beta[0] + beta[1] * data['mom_iq']
        
        # Likelihood
        kid_score_obs = pm.Normal("kid_score", mu=mu, sigma=sigma, observed=data['kid_score'])

    return model