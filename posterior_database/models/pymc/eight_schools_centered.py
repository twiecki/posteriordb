def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        J = data['J']
        y_obs = data['y']
        sigma_data = data['sigma']
        
        # Parameters
        # mu ~ normal(0, 5)
        mu = pm.Normal("mu", mu=0, sigma=5)
        
        # tau ~ cauchy(0, 5) with lower bound 0
        tau = pm.HalfCauchy("tau", beta=5)
        
        # theta ~ normal(mu, tau) - hierarchical parameters
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=J)
        
        # Likelihood: y ~ normal(theta, sigma)
        y = pm.Normal("y", mu=theta, sigma=sigma_data, observed=y_obs)
        
        
    return model