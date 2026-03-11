def make_model(data: dict) -> "pm.Model":
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Parameters with beta(1,1) priors (uniform on [0,1])
        theta1 = pm.Beta("theta1", alpha=1, beta=1)
        theta2 = pm.Beta("theta2", alpha=1, beta=1)
        
        # Transformed parameter: difference between rates
        delta = pm.Deterministic("delta", theta1 - theta2)
        
        # Observed counts with binomial likelihoods
        k1_obs = pm.Binomial("k1", n=data['n1'], p=theta1, observed=data['k1'])
        k2_obs = pm.Binomial("k2", n=data['n2'], p=theta2, observed=data['k2'])

    return model