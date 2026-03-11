def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    K = data['K']
    T = data['T']
    y_data = data['y']
    
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=K)
        sigma = pm.HalfCauchy("sigma", beta=2.5)
        
        # Autoregressive structure for t in (K+1):T (Stan 1-based indexing)
        # In Python 0-based: t in K:(T)
        
        # Build the means for each observation from K onwards
        mu_list = []
        for t in range(K, T):  # t from K to T-1 (0-based)
            mu_t = alpha
            for k in range(K):  # k from 0 to K-1
                mu_t = mu_t + beta[k] * y_data[t - k - 1]  # y[t-k] in Stan (1-based) -> y_data[t-k-1] in Python
            mu_list.append(mu_t)
        
        mu = pt.stack(mu_list)
        
        # Likelihood for observations from K+1 to T (Stan 1-based) -> K to T-1 (0-based)
        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y_data[K:])
        
        # Correction for HalfCauchy log(2) offset to match Stan
        pm.Potential("half_dist_correction", -pt.log(2.0))
    
    return model