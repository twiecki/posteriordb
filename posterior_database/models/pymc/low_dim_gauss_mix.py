def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Data
        N = data['N']
        y_data = data['y']
        
        # Parameters
        # ordered[2] mu - use ordered transform
        mu = pm.Normal("mu", mu=0, sigma=2, shape=2,
                       transform=pm.distributions.transforms.ordered,
                       initval=np.array([-1.0, 1.0]))

        # array[2] real<lower=0> sigma - use HalfNormal since prior is normal(0, 2) with lower=0
        sigma = pm.HalfNormal("sigma", sigma=2, shape=2)

        # real<lower=0, upper=1> theta - mixing probability
        theta = pm.Beta("theta", alpha=5, beta=5)
        
        # Likelihood: mixture of two normals
        # Use pm.NormalMixture for the mixture model
        w = pt.stack([theta, 1 - theta])  # mixing weights
        mu_components = pt.stack([mu[0], mu[1]])
        sigma_components = pt.stack([sigma[0], sigma[1]])
        
        y_obs = pm.NormalMixture("y", w=w, mu=mu_components, sigma=sigma_components, observed=y_data)
        
    return model