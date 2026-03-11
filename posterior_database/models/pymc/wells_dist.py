def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        N = data['N']
        switched = data['switched']
        dist = data['dist']
        
        # Parameters - using flat priors since Stan has no explicit priors
        beta = pm.Flat("beta", shape=2)
        
        # Linear predictor (logit scale)
        logit_p = beta[0] + beta[1] * dist
        
        # Likelihood
        switched_obs = pm.Bernoulli("switched", logit_p=logit_p, observed=switched)

    return model