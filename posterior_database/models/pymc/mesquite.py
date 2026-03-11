def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        N = data['N']
        weight = data['weight']
        diam1 = data['diam1']
        diam2 = data['diam2']
        canopy_height = data['canopy_height']
        total_height = data['total_height']
        density = data['density']
        group = data['group']
        
        # Parameters
        # Stan has vector[7] beta with no explicit prior, so using Flat
        beta = pm.Flat("beta", shape=7)
        
        # real<lower=0> sigma with no explicit prior
        sigma = pm.HalfFlat("sigma")
        
        # Model: linear combination
        mu = (beta[0] + beta[1] * diam1 + beta[2] * diam2 + 
              beta[3] * canopy_height + beta[4] * total_height + 
              beta[5] * density + beta[6] * group)
        
        # Likelihood
        weight_obs = pm.Normal("weight", mu=mu, sigma=sigma, observed=weight)

    return model