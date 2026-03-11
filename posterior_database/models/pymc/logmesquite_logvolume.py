def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract and ensure data is numpy arrays
    weight = np.asarray(data['weight'], dtype=float)
    diam1 = np.asarray(data['diam1'], dtype=float)  
    diam2 = np.asarray(data['diam2'], dtype=float)
    canopy_height = np.asarray(data['canopy_height'], dtype=float)
    
    # Transformed data
    log_weight = np.log(weight)
    log_canopy_volume = np.log(diam1 * diam2 * canopy_height)

    with pm.Model() as model:
        # Parameters - Stan has no explicit priors, so using flat priors
        beta = pm.Flat("beta", shape=2)
        sigma = pm.HalfFlat("sigma")
        
        # Linear model: beta[1] + beta[2] * log_canopy_volume  
        # Note: Stan indexing beta[1], beta[2] -> Python beta[0], beta[1]
        mu = beta[0] + beta[1] * log_canopy_volume
        
        # Likelihood
        pm.Normal("log_weight", mu=mu, sigma=sigma, observed=log_weight)

    return model