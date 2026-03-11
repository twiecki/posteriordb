def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Prior Distribution for Rate Theta
        # Stan: theta ~ beta(1, 1) with bounds [0,1]
        theta = pm.Beta("theta", alpha=1, beta=1)
        
        # Observed Counts
        # Stan: k ~ binomial(n, theta)
        k_obs = pm.Binomial("k", n=data['n'], p=theta, observed=data['k'])

    return model