def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    N = data['N']
    J = data['J']
    county = np.array(data['county']) - 1
    y_obs = np.array(data['y'])

    with pm.Model() as model:
        
        # Parameters with bounds - use Uniform since no explicit priors given
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)  # explicit prior in Stan
        sigma_a = pm.Uniform("sigma_a", lower=0, upper=100)  # bounded, no explicit prior
        sigma_y = pm.Uniform("sigma_y", lower=0, upper=100)  # bounded, no explicit prior
        
        # County-level random effects
        a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=J)
        
        # Likelihood - vectorized using advanced indexing
        y_hat = a[county]
        y_likelihood = pm.Normal("y", mu=y_hat, sigma=sigma_y, observed=y_obs)

    return model