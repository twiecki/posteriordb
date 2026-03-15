def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    nobs = data['nobs']
    nmis = data['nmis']
    nyear = data['nyear']
    nsite = data['nsite']
    obs = np.array(data['obs'])
    obsyear = np.array(data['obsyear']) - 1
    obssite = np.array(data['obssite']) - 1
    misyear = np.array(data['misyear']) - 1
    missite = np.array(data['missite']) - 1

    with pm.Model() as model:
        
        # Parameters
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        sd_alpha = pm.Uniform("sd_alpha", lower=0, upper=5)  # Bounded parameter
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sd_alpha, shape=nsite)
        
        # Transformed parameters
        # log_lambda = rep_matrix(alpha', nyear) means replicate alpha across years
        # In PyMC, we can use broadcasting or indexing
        log_lambda = pm.Deterministic("log_lambda", 
                                     pt.tile(alpha[None, :], (nyear, 1)))
        
        # Model - likelihood for observed data
        # obs[i] ~ poisson_log(log_lambda[obsyear[i], obssite[i]])
        pm.Poisson("obs", mu=pt.exp(log_lambda[obsyear, obssite]), observed=obs)
        
    return model