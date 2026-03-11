def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    with pm.Model() as model:
        # Parameters
        beta = pm.Flat("beta", shape=2)  # vector[2] beta with no explicit prior
        sigma = pm.HalfCauchy("sigma", beta=2.5)  # real<lower=0> sigma ~ cauchy(0, 2.5)
        
        # Model
        mu = beta[0] + beta[1] * data["mom_hs"]
        kid_score_obs = pm.Normal("kid_score", mu=mu, sigma=sigma, observed=data["kid_score"])
        
    return model