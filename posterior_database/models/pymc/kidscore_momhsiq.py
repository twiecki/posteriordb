def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Parameters
        beta = pm.Flat("beta", shape=3)  # No prior specified in Stan = improper uniform
        sigma = pm.HalfCauchy("sigma", beta=2.5)  # cauchy(0, 2.5) with lower=0 constraint
        
        # Linear predictor
        mu = beta[0] + beta[1] * data["mom_hs"] + beta[2] * data["mom_iq"]
        
        # Likelihood
        kid_score_obs = pm.Normal("kid_score", mu=mu, sigma=sigma, observed=data["kid_score"])

    return model