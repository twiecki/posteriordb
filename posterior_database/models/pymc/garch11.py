def make_model(data: dict) -> pm.Model:
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
        
        # Add Jacobian for the transform: log|d(beta1)/d(beta1_raw)| = log(1 - alpha1)
        pm.Potential("beta1_jacobian", pt.log(1 - alpha1))
        
        # Build sigma array manually since T is known
        sigma_list = [sigma1]  # sigma[0] = sigma1 (corresponds to sigma[1] in Stan)
        
        for t in range(1, T):  # t=1 to T-1 (corresponds to t=2 to T in Stan)
            # In Stan: sigma[t] = sqrt(alpha0 + alpha1 * square(y[t-1] - mu) + beta1 * square(sigma[t-1]))
            # With 0-based indexing: sigma[t] uses y[t-1] and sigma[t-1]
            sigma_prev = sigma_list[t-1]
            y_prev = y[t-1]
            sigma_curr = pt.sqrt(alpha0 + alpha1 * (y_prev - mu)**2 + beta1 * sigma_prev**2)
            sigma_list.append(sigma_curr)
        
        sigma = pt.stack(sigma_list)
        sigma = pm.Deterministic("sigma", sigma)
        
        # Likelihood
        y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        
        # Correction for half distributions (alpha0 uses HalfFlat)
        pm.Potential("half_dist_correction", -1 * pt.log(2.0))

    return model