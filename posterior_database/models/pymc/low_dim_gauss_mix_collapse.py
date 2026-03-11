def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        N = data['N']
        y = data['y']
        
        # Parameters
        # mu is a 2D vector with normal prior
        mu = pm.Normal("mu", mu=0, sigma=2, shape=2)
        
        # sigma is array of 2 positive reals with half-normal prior
        # Stan: sigma ~ normal(0, 2) with <lower=0> constraint
        sigma = pm.HalfNormal("sigma", sigma=2, shape=2)
        
        # theta is mixing proportion with beta prior
        theta = pm.Beta("theta", alpha=5, beta=5)
        
        # Mixture model likelihood
        # Stan uses log_mix(theta, normal_lpdf(y[n] | mu[1], sigma[1]), normal_lpdf(y[n] | mu[2], sigma[2]))
        # In PyMC, we can use pm.Mixture
        components = [
            pm.Normal.dist(mu=mu[0], sigma=sigma[0]),
            pm.Normal.dist(mu=mu[1], sigma=sigma[1])
        ]
        
        # Mixing weights: theta is probability of first component, (1-theta) is probability of second
        weights = pt.stack([theta, 1 - theta])
        
        y_obs = pm.Mixture("y", w=weights, comp_dists=components, observed=y)

    return model