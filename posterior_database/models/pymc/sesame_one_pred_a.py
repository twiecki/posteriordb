def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Parameters
        # Stan: vector[2] beta (no explicit prior = improper uniform)
        beta = pm.Flat("beta", shape=2)
        
        # Stan: real<lower=0> sigma (no explicit prior = improper half-flat)  
        sigma = pm.HalfFlat("sigma")
        
        # Linear predictor
        mu = beta[0] + beta[1] * np.array(data['encouraged'])
        
        # Likelihood
        # Stan: watched ~ normal(beta[1] + beta[2] * encouraged, sigma)
        # Note: Stan uses 1-based indexing, so beta[1] = beta[0] in Python
        # Convert watched to float to match Stan's vector type
        watched_data = np.array(data['watched'], dtype=float)
        watched_obs = pm.Normal("watched", mu=mu, sigma=sigma, observed=watched_data)
        
        # Remove normalization constant to match Stan's propto=True behavior
        # Each normal contributes -0.5 * log(2*pi) which Stan drops
        N = len(watched_data)
        pm.Potential("normalization_correction", N * 0.5 * pt.log(2.0 * np.pi))

    return model