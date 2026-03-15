def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Data
        N = data['N']
        y_data = data['y']
        
        # Parameters
        theta = pm.Uniform('theta', lower=0, upper=1)
        mu = pm.Normal('mu', mu=0, sigma=10, shape=2)
        
        # Mixture model likelihood
        # Use pm.Mixture to handle the mixture likelihood
        components = [
            pm.Normal.dist(mu=mu[0], sigma=1.0),
            pm.Normal.dist(mu=mu[1], sigma=1.0)
        ]
        
        weights = pt.stack([theta, 1 - theta])
        
        y_obs = pm.Mixture('y', w=weights, comp_dists=components, observed=y_data)

    return model