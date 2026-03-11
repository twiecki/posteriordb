def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Convert 1-based county indices to 0-based
    county_idx = np.array(data['county_idx']) - 1
    floor_measure = np.array(data['floor_measure'])
    log_radon = np.array(data['log_radon'])
    
    with pm.Model() as model:
        # Parameters
        alpha = pm.Normal("alpha", mu=0, sigma=10, shape=data['J'])
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1)
        
        # Linear predictor
        mu = alpha[county_idx] + beta * floor_measure
        
        # Likelihood
        log_radon_obs = pm.Normal("log_radon", mu=mu, sigma=sigma_y, observed=log_radon)

    return model