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
    y_sd = np.std(y)  # ddof=0 like Stan's sd()
    mu_lower = y_mean - 3 * y_sd
    mu_upper = y_mean + 3 * y_sd
    
    with pm.Model() as model:
        # Parameters with explicit priors where needed
        # mu with bounds - implicit flat priors in Stan for unconstrained parts
        mu = pm.Uniform("mu", lower=mu_lower, upper=mu_upper, shape=n)
        
        # seasonal components - implicit flat priors in Stan  
        seasonal = pm.Flat("seasonal", shape=n)
        
        # Regression coefficients - implicit flat priors in Stan
        beta = pm.Flat("beta")
        lambda_ = pm.Flat("lambda")
        
        # For positive_ordered, use a different approach
        # Start with unconstrained variables for the ordered part
        sigma_raw = pm.Normal("sigma_raw", mu=0, sigma=1, shape=3)
        
        # Transform to get positive ordered values
        # sigma[0] = exp(sigma_raw[0])
        # sigma[1] = sigma[0] + exp(sigma_raw[1])  
        # sigma[2] = sigma[1] + exp(sigma_raw[2])
        sigma_0 = pt.exp(sigma_raw[0])
        sigma_1 = sigma_0 + pt.exp(sigma_raw[1])
        sigma_2 = sigma_1 + pt.exp(sigma_raw[2])
        sigma = pt.stack([sigma_0, sigma_1, sigma_2])
        sigma = pm.Deterministic("sigma", sigma)
        
        # Add prior for sigma ~ student_t(4, 0, 1)
        for i in range(3):
            pm.Potential(f"sigma_prior_{i}", 
                        pm.logp(pm.StudentT.dist(nu=4, mu=0, sigma=1), sigma[i]))
        
        # Seasonal constraints: for t >= 12, seasonal[t] ~ normal(-sum(seasonal[t-11:t-1]), sigma[0])
        for t in range(11, n):  # t in 12:n (Stan) -> t in 11:n (Python 0-based)
            seasonal_sum = pt.sum(seasonal[t-11:t])  # sum from t-11 to t-1 (inclusive)
            pm.Potential(f"seasonal_constraint_{t}", 
                        pm.logp(pm.Normal.dist(mu=-seasonal_sum, sigma=sigma[0]), seasonal[t]))
        
        # Random walk for mu: for t >= 2, mu[t] ~ normal(mu[t-1], sigma[1])
        for t in range(1, n):  # t in 2:n (Stan) -> t in 1:n (Python 0-based)
            pm.Potential(f"mu_constraint_{t}", 
                        pm.logp(pm.Normal.dist(mu=mu[t-1], sigma=sigma[1]), mu[t]))
        
        # Transformed parameter
        yhat = mu + beta * x + lambda_ * w
        
        # Likelihood
        y_obs = pm.Normal("y", mu=yhat + seasonal, sigma=sigma[2], observed=y)
        
    return model