def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data and convert to numpy arrays
    N = data['N']
    weight = np.array(data['weight'])
    diam1 = np.array(data['diam1'])
    diam2 = np.array(data['diam2'])
    canopy_height = np.array(data['canopy_height'])
    total_height = np.array(data['total_height'])
    group = np.array(data['group'])
    
    # Transformed data (computed before model)
    log_weight = np.log(weight)
    log_canopy_volume = np.log(diam1 * diam2 * canopy_height)
    log_canopy_area = np.log(diam1 * diam2)
    log_canopy_shape = np.log(diam1 / diam2)
    log_total_height = np.log(total_height)

    with pm.Model() as model:
        # Parameters with improper uniform priors (no explicit priors in Stan)
        beta = pm.Flat("beta", shape=6)
        
        # For sigma: real<lower=0> with no explicit prior = improper uniform on (0,∞)
        # Use log-transform to match Stan's parameterization
        sigma_log = pm.Flat("sigma_log")
        sigma = pm.Deterministic("sigma", pt.exp(sigma_log))
        
        # Model - Stan uses 1-based indexing, PyMC uses 0-based
        mu = (beta[0] + beta[1] * log_canopy_volume + 
              beta[2] * log_canopy_area + 
              beta[3] * log_canopy_shape + 
              beta[4] * log_total_height + 
              beta[5] * group)
        
        # The Normal likelihood
        log_weight_obs = pm.Normal("log_weight", mu=mu, sigma=sigma, observed=log_weight)
        

    return model