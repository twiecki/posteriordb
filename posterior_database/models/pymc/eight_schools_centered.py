def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        J = data['J']
        y = data['y']
        sigma = data['sigma']
        
        # Parameters
        # tau ~ cauchy(0, 5) with lower bound 0 -> HalfCauchy
        tau = pm.HalfCauchy("tau", beta=5)
        
        # mu ~ normal(0, 5)
        mu = pm.Normal("mu", mu=0, sigma=5)
        
        # theta ~ normal(mu, tau) - hierarchical parameters
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=J)
        
        # y ~ normal(theta, sigma) - likelihood
        y_obs = pm.Normal("y", mu=theta, sigma=sigma, observed=y)

    return model