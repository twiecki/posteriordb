def make_model(data: dict):
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    K = data['K']
    T = data['T'] 
    y = data['y']
    
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=K)
        sigma = pm.HalfCauchy("sigma", beta=2.5)
        
        # For AR model, we need to compute the mean for observations from K+1 to T
        # In Python indexing: from K to T (since we use 0-based indexing)
        
        # Create lagged predictors matrix: each row has y[t-K], y[t-K+1], ..., y[t-1]
        # For time t (0-based), we need y[t-K], y[t-K+1], ..., y[t-1]
        lagged_y = []
        for t in range(K, T):
            lag_vector = []
            for k in range(1, K + 1):  # k goes from 1 to K (Stan indexing)
                lag_vector.append(y[t - k])  # y[t-k] in 0-based indexing
            lagged_y.append(lag_vector)
        
        lagged_y = np.array(lagged_y)  # Shape: (T-K, K)
        
        # Compute mean for each time point
        mu = alpha + pt.dot(lagged_y, beta)
        
        # Likelihood for observations from time K+1 onwards (0-based: K to T-1)
        y_obs = y[K:T]  # Observed values from time K to T-1 (0-based)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

    return model