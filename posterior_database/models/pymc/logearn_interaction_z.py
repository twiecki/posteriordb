def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Transformed data computations (same as Stan's transformed data block)
    log_earn = np.log(data['earn'])
    z_height = (data['height'] - np.mean(data['height'])) / np.std(data['height'])
    inter = z_height * data['male']
    
    with pm.Model() as model:
        # Parameters
        beta = pm.Flat("beta", shape=4)  # No explicit prior in Stan = improper uniform
        sigma = pm.HalfFlat("sigma")     # real<lower=0> with no explicit prior
        
        # Model: linear combination
        mu = beta[0] + beta[1] * z_height + beta[2] * data['male'] + beta[3] * inter
        
        # Likelihood
        log_earn_obs = pm.Normal("log_earn", mu=mu, sigma=sigma, observed=log_earn)

    return model