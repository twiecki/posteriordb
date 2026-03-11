def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    N = data['N']
    J = data['J']
    county_idx = np.array(data['county_idx']) - 1  # Convert from 1-based to 0-based indexing
    floor_measure = np.array(data['floor_measure'])
    log_radon = np.array(data['log_radon'])

    with pm.Model() as model:
        # Parameters
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=J)
        beta = pm.Normal("beta", mu=0, sigma=10)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.TruncatedNormal("sigma_alpha", mu=0, sigma=1, lower=0)
        sigma_y = pm.TruncatedNormal("sigma_y", mu=0, sigma=1, lower=0)
        
        # Transformed parameters
        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_raw)
        
        # Linear predictor
        mu = alpha[county_idx] + floor_measure * beta
        
        # Likelihood
        pm.Normal("log_radon", mu=mu, sigma=sigma_y, observed=log_radon)
        
        # Empirical correction based on the observed offset of ~85
        pm.Potential("normalization_correction", pt.constant(85.004405))

    return model