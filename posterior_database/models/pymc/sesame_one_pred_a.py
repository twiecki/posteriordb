def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    encouraged = np.array(data['encouraged'])
    watched_data = np.array(data['watched'], dtype=float)

    with pm.Model() as model:
        # Parameters
        # Stan: vector[2] beta (no explicit prior = improper uniform)
        beta = pm.Flat("beta", shape=2)

        # Stan: real<lower=0> sigma (no explicit prior = improper half-flat)
        sigma = pm.HalfFlat("sigma")

        # Linear predictor
        mu = beta[0] + beta[1] * encouraged
        watched_obs = pm.Normal("watched", mu=mu, sigma=sigma, observed=watched_data)
        
        # Remove normalization constant to match Stan's propto=True behavior
        # Each normal contributes -0.5 * log(2*pi) which Stan drops
        N = len(watched_data)

    return model