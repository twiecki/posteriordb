def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        N = data['N']
        x = data['x']
        y = data['y']
        xpred = data['xpred']
        pmualpha = data['pmualpha']
        psalpha = data['psalpha']
        pmubeta = data['pmubeta']
        psbeta = data['psbeta']
        
        # Parameters with priors
        alpha = pm.Normal("alpha", mu=pmualpha, sigma=psalpha)
        beta = pm.Normal("beta", mu=pmubeta, sigma=psbeta)
        sigma = pm.HalfFlat("sigma")  # real<lower=0> with no explicit prior
        
        # Likelihood
        mu = alpha + beta * x
        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    return model