def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Get data
        N = data['N']
        J = data['J']
        county_idx = np.array(data['county_idx']) - 1  # Convert to 0-based indexing
        log_uppm = data['log_uppm']
        floor_measure = data['floor_measure']
        log_radon = data['log_radon']
        
        # Priors
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
        
        # County-level random effects
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=J)
        
        # Linear predictor
        # mu[n] = alpha[county_idx[n]] + log_uppm[n] * beta[1] + floor_measure[n] * beta[2]
        mu = alpha[county_idx] + log_uppm * beta[0] + floor_measure * beta[1]
        
        # Likelihood
        y_obs = pm.Normal("log_radon", mu=mu, sigma=sigma_y, observed=log_radon)

    return model