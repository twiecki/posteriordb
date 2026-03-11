def make_model(data: dict):
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    T = data['T']
    y = data['y']
    sigma1 = data['sigma1']
    
    with pm.Model() as model:
        # Parameters
        mu = pm.Flat("mu")
        alpha0 = pm.HalfFlat("alpha0")
        alpha1 = pm.Uniform("alpha1", lower=0, upper=1)
        
        # beta1 has dependent bounds: 0 < beta1 < (1 - alpha1)
        # Use a helper variable and transform
        beta1_raw = pm.Uniform("beta1_raw", lower=0, upper=1)
        beta1 = pm.Deterministic("beta1", beta1_raw * (1 - alpha1))
        
        # Add Jacobian for the transformation: log|d(beta1)/d(beta1_raw)| = log(1 - alpha1)
        pm.Potential("beta1_jacobian", pt.log(1 - alpha1))
        
        # Compute time-varying sigma recursively
        def compute_sigma(y_vals, mu_val, alpha0_val, alpha1_val, beta1_val, sigma1_val):
            sigma = pt.zeros(T)
            sigma = pt.set_subtensor(sigma[0], sigma1_val)
            
            for t in range(1, T):
                sigma_t_squared = (alpha0_val + 
                                 alpha1_val * (y_vals[t-1] - mu_val)**2 + 
                                 beta1_val * sigma[t-1]**2)
                sigma = pt.set_subtensor(sigma[t], pt.sqrt(sigma_t_squared))
            
            return sigma
        
        sigma = compute_sigma(y, mu, alpha0, alpha1, beta1, sigma1)
        
        # Likelihood
        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    
    return model