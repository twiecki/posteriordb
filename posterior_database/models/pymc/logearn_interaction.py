def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data and convert to numpy arrays
    N = data['N']
    earn = np.array(data['earn'], dtype=float)
    height = np.array(data['height'], dtype=float)
    male = np.array(data['male'], dtype=float)
    
    # Transformed data (same as Stan's transformed data block)
    log_earn = np.log(earn)
    inter = height * male  # element-wise multiplication
    
    with pm.Model() as model:
        # Parameters
        beta = pm.Flat("beta", shape=4)  # No explicit prior in Stan = improper uniform
        sigma = pm.HalfFlat("sigma")     # real<lower=0> with no explicit prior
        
        # Model: linear combination
        mu = beta[0] + beta[1] * height + beta[2] * male + beta[3] * inter
        
        # Likelihood
        log_earn_obs = pm.Normal("log_earn", mu=mu, sigma=sigma, observed=log_earn)
    
    return model