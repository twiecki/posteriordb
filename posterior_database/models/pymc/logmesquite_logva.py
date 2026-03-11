def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Convert to numpy arrays to ensure proper arithmetic
    weight = np.array(data['weight'])
    diam1 = np.array(data['diam1'])
    diam2 = np.array(data['diam2'])
    canopy_height = np.array(data['canopy_height'])
    group = np.array(data['group'])
    
    # Transformed data - compute derived quantities from data
    log_weight = np.log(weight)
    log_canopy_volume = np.log(diam1 * diam2 * canopy_height)
    log_canopy_area = np.log(diam1 * diam2)
    
    with pm.Model() as model:
        # Parameters - no explicit priors in Stan means improper uniform
        beta = pm.Flat("beta", shape=4)
        sigma = pm.HalfFlat("sigma")
        
        # Linear predictor
        mu = (beta[0] + 
              beta[1] * log_canopy_volume + 
              beta[2] * log_canopy_area + 
              beta[3] * group)
        
        # Likelihood
        log_weight_obs = pm.Normal("log_weight", mu=mu, sigma=sigma, observed=log_weight)
        
    return model