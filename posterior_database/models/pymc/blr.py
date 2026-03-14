def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        N = data['N']
        D = data['D'] 
        X = data['X']
        y = data['y']
        
        # Parameters
        beta = pm.Normal("beta", mu=0, sigma=10, shape=D)
        sigma = pm.HalfNormal("sigma", sigma=10)
        
        # Linear predictor
        mu = X @ beta
        
        # Likelihood
        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    return model