def make_model(data: dict, prior_only: bool = False) -> pm.Model:
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        N = data['N']
        y = data['y']
        
        mu = pm.Normal("mu", mu=0, sigma=2, shape=2)
        sigma = pm.HalfNormal("sigma", sigma=2, shape=2)
        theta = pm.Beta("theta", alpha=5, beta=5)
        
        components = [
            pm.Normal.dist(mu=mu[0], sigma=sigma[0]),
            pm.Normal.dist(mu=mu[1], sigma=sigma[1])
        ]
        
        weights = pt.stack([theta, 1 - theta])
        
        if not prior_only:
            y_obs = pm.Mixture("y", w=weights, comp_dists=components, observed=y)

    return model