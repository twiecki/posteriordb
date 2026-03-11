def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    n = data['n']
    y = data['y']
    x = data['x']
    w = data['w']
    
    # Compute bounds for mu based on data statistics
    y_mean = np.mean(y)
    y_sd = np.std(y)  # ddof=0 by default, matches Stan's sd()
    mu_lower = y_mean - 3 * y_sd
    mu_upper = y_mean + 3 * y_sd

    with pm.Model() as model:
        # Parameters
        # mu with bounds based on data statistics
        mu = pm.Uniform("mu", lower=mu_lower, upper=mu_upper, shape=n)
        
        # seasonal component (unbounded)
        seasonal = pm.Flat("seasonal", shape=n)
        
        # beta and lambda (unbounded scalars)
        beta = pm.Flat("beta")
        lambda_ = pm.Flat("lambda")
        
        # positive_ordered sigma with student_t prior
        # Use a manual parameterization: sigma[0], sigma[1] - sigma[0], sigma[2] - sigma[1]
        sigma_0 = pm.StudentT("sigma_0", nu=4, mu=0, sigma=1, transform=pm.distributions.transforms.log)
        sigma_diff_1 = pm.StudentT("sigma_diff_1", nu=4, mu=0, sigma=1, transform=pm.distributions.transforms.log) 
        sigma_diff_2 = pm.StudentT("sigma_diff_2", nu=4, mu=0, sigma=1, transform=pm.distributions.transforms.log)
        
        sigma = pm.Deterministic("sigma", pt.stack([
            sigma_0,
            sigma_0 + sigma_diff_1,
            sigma_0 + sigma_diff_1 + sigma_diff_2
        ]))
        
        # Transformed parameters
        yhat = pm.Deterministic("yhat", mu + beta * x + lambda_ * w)
        
        # Model constraints using Potential
        # Seasonal constraint: for t in 12:n, seasonal[t] ~ normal(-sum(seasonal[t-11:t-1]), sigma[1])
        # Stan t=12 means index 11 in Python (0-based)
        # seasonal[t-11:t-1] in Stan means seasonal[(t-11):(t-1)] = seasonal[t-11], ..., seasonal[t-1]
        # In Python 0-based, this is seasonal[t-1-11:t-1] = seasonal[t-12:t-1]
        seasonal_logp = 0
        for t in range(11, n):  # Stan 12:n becomes Python 11:n (0-based)
            seasonal_sum = pt.sum(seasonal[t-11:t])  # This sums seasonal[t-11] through seasonal[t-1]
            seasonal_logp += pm.Normal.logp(seasonal[t], mu=-seasonal_sum, sigma=sigma[0])
        pm.Potential("seasonal_constraint", seasonal_logp)
        
        # Random walk for mu: for t in 2:n, mu[t] ~ normal(mu[t-1], sigma[2])
        mu_logp = 0
        for t in range(1, n):  # Stan 2:n becomes Python 1:n (0-based)
            mu_logp += pm.Normal.logp(mu[t], mu=mu[t-1], sigma=sigma[1])
        pm.Potential("mu_constraint", mu_logp)
        
        # Main likelihood: y ~ normal(yhat + seasonal, sigma[3])
        y_obs = pm.Normal("y", mu=yhat + seasonal, sigma=sigma[2], observed=y)

    return model