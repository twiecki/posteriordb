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
        theta_trans = pm.Normal("theta_trans", mu=0, sigma=1, shape=J)
        mu = pm.Normal("mu", mu=0, sigma=5)
        tau = pm.HalfCauchy("tau", beta=5)  # tau ~ cauchy(0, 5) with lower=0 constraint
        
        # Transformed parameters
        theta = pm.Deterministic("theta", theta_trans * tau + mu)
        
        # Likelihood
        y_obs = pm.Normal("y", mu=theta, sigma=sigma, observed=y)

    return model