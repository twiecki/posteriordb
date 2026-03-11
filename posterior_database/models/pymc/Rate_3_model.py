def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Prior on theta - beta(1,1) is uniform on [0,1]
        theta = pm.Beta("theta", alpha=1, beta=1)
        
        # Observed counts - binomial likelihood
        k1_obs = pm.Binomial("k1", n=data['n1'], p=theta, observed=data['k1'])
        k2_obs = pm.Binomial("k2", n=data['n2'], p=theta, observed=data['k2'])

    return model