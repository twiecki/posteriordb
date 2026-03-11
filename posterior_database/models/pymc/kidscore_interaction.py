def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data and convert to numpy arrays
    N = data['N']
    kid_score = np.array(data['kid_score'])
    mom_iq = np.array(data['mom_iq']) 
    mom_hs = np.array(data['mom_hs'])
    
    # Transformed data: interaction term
    inter = mom_hs * mom_iq
    
    with pm.Model() as model:
        # Parameters
        beta = pm.Flat("beta", shape=4)  # No explicit priors in Stan = improper uniform
        sigma = pm.HalfCauchy("sigma", beta=2.5)  # Cauchy(0, 2.5) with lower=0 constraint
        
        # Linear model
        mu = beta[0] + beta[1] * mom_hs + beta[2] * mom_iq + beta[3] * inter
        
        # Likelihood
        y_obs = pm.Normal("kid_score", mu=mu, sigma=sigma, observed=kid_score)
        
    return model