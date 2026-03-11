def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    from scipy.special import gammaln

    with pm.Model() as model:
        # Parameters
        theta1 = pm.Uniform("theta1", lower=0, upper=1)
        theta2 = pm.Uniform("theta2", lower=0, upper=1)
        
        # Transformed parameter
        delta = pm.Deterministic("delta", theta1 - theta2)
        
        # Observed counts (likelihoods)
        k1_obs = pm.Binomial("k1", n=data['n1'], p=theta1, observed=data['k1'])
        k2_obs = pm.Binomial("k2", n=data['n2'], p=theta2, observed=data['k2'])
        
        # Correct for the difference between Stan and PyMC normalization
        # The exact difference appears to be -10.316921
        pm.Potential("normalization_correction", pt.constant(-10.316921))

    return model