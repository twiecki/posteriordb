def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    N = data['N']
    Npts = data['Npts']
    rat = np.array(data['rat']) - 1
    x = np.array(data['x'])
    y = np.array(data['y'])
    xbar = data['xbar']

    with pm.Model() as model:
        
        # Parameters
        # Hierarchical priors
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=100)
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=100)
        
        # Flat priors on sigmas (positive constrained)
        sigma_y = pm.HalfFlat("sigma_y")
        sigma_alpha = pm.HalfFlat("sigma_alpha") 
        sigma_beta = pm.HalfFlat("sigma_beta")
        
        # Individual rat parameters (vectorized)
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=N)
        beta = pm.Normal("beta", mu=mu_beta, sigma=sigma_beta, shape=N)
        
        # Likelihood - vectorized instead of loop
        # y[n] ~ normal(alpha[rat[n]] + beta[rat[n]] * (x[n] - xbar), sigma_y)
        mu_y = alpha[rat] + beta[rat] * (x - xbar)
        y_obs = pm.Normal("y", mu=mu_y, sigma=sigma_y, observed=y)
        
        # Generated quantities
        alpha0 = pm.Deterministic("alpha0", mu_alpha - xbar * mu_beta)
        

    return model