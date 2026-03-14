def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    N = data['N']
    earn = data['earn']
    height = data['height']
    male = data['male']
    
    # Transformed data (computed before model definition)
    log_earn = np.log(earn)
    z_height = (height - np.mean(height)) / np.std(height)  # Stan's sd() uses ddof=0
    inter = z_height * male
    
    with pm.Model() as model:
        # Parameters - no explicit priors in Stan means improper uniform
        beta = pm.Flat("beta", shape=4)
        
        # real<lower=0> sigma with no explicit prior
        sigma = pm.HalfFlat("sigma")
        
        
        
        # Model
        mu = beta[0] + beta[1] * z_height + beta[2] * male + beta[3] * inter
        log_earn_obs = pm.Normal("log_earn", mu=mu, sigma=sigma, observed=log_earn)
        
    return model