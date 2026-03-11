def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    N = data['N']
    J = data['J']
    county_idx = np.array(data['county_idx']) - 1  # Convert to 0-based indexing
    log_uppm = np.array(data['log_uppm'])
    floor_measure = np.array(data['floor_measure'])
    log_radon = np.array(data['log_radon'])

    with pm.Model() as model:
        # Parameters
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=J)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
        sigma_y = pm.HalfNormal("sigma_y", sigma=1)
        
        # Transformed parameters
        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_raw)
        
        # Model likelihood - vectorized
        muj = alpha[county_idx] + log_uppm * beta[0]
        mu = muj + floor_measure * beta[1]
        
        log_radon_obs = pm.Normal("log_radon", mu=mu, sigma=sigma_y, observed=log_radon)

    return model