def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Data
        N = data['N']
        r = data['r']
        n = data['n']
        
        # Parameters
        mu = pm.Normal("mu", mu=0.0, sigma=1000.0)
        sigmasq = pm.InverseGamma("sigmasq", alpha=0.001, beta=0.001)
        
        # Transformed parameter: sigma = sqrt(sigmasq)
        sigma = pm.Deterministic("sigma", pt.sqrt(sigmasq))
        
        # Hierarchical parameters
        b = pm.Normal("b", mu=mu, sigma=sigma, shape=N)
        
        # Transformed parameter: p = inv_logit(b)
        p = pm.Deterministic("p", pm.math.invlogit(b))
        
        # Likelihood using binomial_logit
        r_obs = pm.Binomial("r", n=n, logit_p=b, observed=r)
        
        # Generated quantity
        pop_mean = pm.Deterministic("pop_mean", pm.math.invlogit(mu))
        
    return model