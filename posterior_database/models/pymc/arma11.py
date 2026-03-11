def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    T = data['T']
    y = data['y']
    
    with pm.Model() as model:
        # Priors
        mu = pm.Normal("mu", mu=0, sigma=10)
        phi = pm.Normal("phi", mu=0, sigma=2)
        theta = pm.Normal("theta", mu=0, sigma=2)
        sigma = pm.HalfCauchy("sigma", beta=2.5)
        
        # Initialize arrays for nu and err
        nu = pt.zeros(T)
        err = pt.zeros(T)
        
        # Initial conditions (t=1, which is index 0 in Python)
        nu_1 = mu + phi * mu  # assume err[0] == 0
        err_1 = y[0] - nu_1
        
        # Build the sequences recursively
        nu = pt.set_subtensor(nu[0], nu_1)
        err = pt.set_subtensor(err[0], err_1)
        
        # Recursive computation for t >= 2 (index >= 1)
        for t in range(1, T):
            nu_t = mu + phi * y[t-1] + theta * err[t-1]
            err_t = y[t] - nu_t
            nu = pt.set_subtensor(nu[t], nu_t)
            err = pt.set_subtensor(err[t], err_t)
        
        # Likelihood: err ~ normal(0, sigma) using Potential
        log_likelihood = pt.sum(pm.logp(pm.Normal.dist(mu=0, sigma=sigma), err))
        pm.Potential("likelihood", log_likelihood)
        
        # Correction for HalfCauchy vs Stan's cauchy(0, 2.5) with <lower=0>
        pm.Potential("half_dist_correction", -pt.log(2.0))
    
    return model