def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    with pm.Model() as model:
        # Extract data dimensions
        I = data['I']  # number of persons
        J = data['J']  # number of items
        y = data['y']  # response matrix [I, J]
        
        # Parameters with priors
        # sigma_theta ~ cauchy(0, 2)
        sigma_theta = pm.HalfCauchy("sigma_theta", beta=2)
        
        # theta ~ normal(0, sigma_theta) - ability parameters
        theta = pm.Normal("theta", mu=0, sigma=sigma_theta, shape=J)
        
        # sigma_a ~ cauchy(0, 2)
        sigma_a = pm.HalfCauchy("sigma_a", beta=2)
        
        # a ~ lognormal(0, sigma_a) - discrimination parameters (positive)
        a = pm.LogNormal("a", mu=0, sigma=sigma_a, shape=I)
        
        # mu_b ~ normal(0, 5)
        mu_b = pm.Normal("mu_b", mu=0, sigma=5)
        
        # sigma_b ~ cauchy(0, 2)
        sigma_b = pm.HalfCauchy("sigma_b", beta=2)
        
        # b ~ normal(mu_b, sigma_b) - difficulty parameters
        b = pm.Normal("b", mu=mu_b, sigma=sigma_b, shape=I)
        
        # Likelihood: y[i] ~ bernoulli_logit(a[i] * (theta - b[i]))
        # Need to broadcast: a[i] is shape (I,), theta is shape (J,), b[i] is shape (I,)
        # We want a[i][:, None] * (theta[None, :] - b[i][:, None])
        # This gives shape (I, J) matching y
        
        a_expanded = a[:, None]  # shape (I, 1)
        b_expanded = b[:, None]  # shape (I, 1)
        theta_expanded = theta[None, :]  # shape (1, J)
        
        logit_p = a_expanded * (theta_expanded - b_expanded)  # shape (I, J)
        
        y_obs = pm.Bernoulli("y", logit_p=logit_p, observed=y)
        
    return model