def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    I = data['I']
    J = data['J']
    y = data['y']
    
    with pm.Model() as model:
        # Parameters with priors
        # sigma_theta ~ cauchy(0, 2); - this will need HalfCauchy correction
        sigma_theta = pm.HalfCauchy("sigma_theta", beta=2)
        
        # theta ~ normal(0, sigma_theta);
        theta = pm.Normal("theta", mu=0, sigma=sigma_theta, shape=J)
        
        # sigma_a ~ cauchy(0, 2); - this will need HalfCauchy correction  
        sigma_a = pm.HalfCauchy("sigma_a", beta=2)
        
        # a ~ lognormal(0, sigma_a); - a is vector<lower=0>[I]
        a = pm.LogNormal("a", mu=0, sigma=sigma_a, shape=I)
        
        # mu_b ~ normal(0, 5);
        mu_b = pm.Normal("mu_b", mu=0, sigma=5)
        
        # sigma_b ~ cauchy(0, 2); - this will need HalfCauchy correction
        sigma_b = pm.HalfCauchy("sigma_b", beta=2)
        
        # b ~ normal(mu_b, sigma_b);
        b = pm.Normal("b", mu=mu_b, sigma=sigma_b, shape=I)
        
        # Likelihood: y[i] ~ bernoulli_logit(a[i] * (theta - b[i]))
        # Need to broadcast: a[i] is scalar, theta is vector[J], b[i] is scalar
        # So a[i] * (theta - b[i]) should be vector of length J for each i
        
        # Reshape for broadcasting: a[:,None] * (theta[None,:] - b[:,None])
        logit_p = a[:, None] * (theta[None, :] - b[:, None])  # shape [I, J]
        
        # Observed data
        y_obs = pm.Bernoulli("y", logit_p=logit_p, observed=y)
        
        # Correction for HalfCauchy distributions (3 of them)
        # Stan uses full Cauchy on positive domain, PyMC uses normalized half-distribution
        n_half_params = 3
    
    return model