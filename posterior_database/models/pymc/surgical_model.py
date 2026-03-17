def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    N = data['N']
    n = data['n']
    r = data['r']

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1000)
        sigmasq = pm.InverseGamma("sigmasq", alpha=0.001, beta=0.001)

        sigma = pm.Deterministic("sigma", pt.sqrt(sigmasq))

        b = pm.Normal("b", mu=mu, sigma=sigma, shape=N)

        p = pm.Deterministic("p", pm.math.invlogit(b))

        pm.Binomial("r_obs", n=n, logit_p=b, observed=r)

        pop_mean = pm.Deterministic("pop_mean", pm.math.invlogit(mu))

    return model
