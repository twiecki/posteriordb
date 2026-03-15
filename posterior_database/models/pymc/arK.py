def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    K = data['K']
    T = data['T']
    y_data = data['y']
    
    y_arr = np.array(y_data)
    lag_matrix = np.column_stack([y_arr[K-k-1:T-k-1] for k in range(K)])

    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=K)
        sigma = pm.HalfCauchy("sigma", beta=2.5)
        mu = alpha + lag_matrix @ beta
        
        # Likelihood for observations from K+1 to T (Stan 1-based) -> K to T-1 (0-based)
        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y_data[K:])
        
    
    return model