def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Parameters with Beta(1,1) priors (uniform on [0,1])
        theta = pm.Beta("theta", alpha=1, beta=1)
        thetaprior = pm.Beta("thetaprior", alpha=1, beta=1)
        
        # Observed data - binomial likelihood
        k_obs = pm.Binomial("k", n=data['n'], p=theta, observed=data['k'])
        
        # Generated quantities are NOT part of the model logp
        # They would be computed during posterior sampling if needed
        # but don't contribute to the likelihood

    return model