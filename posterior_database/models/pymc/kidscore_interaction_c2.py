def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data and convert to numpy arrays
    N = data['N']
    kid_score = np.array(data['kid_score'])
    mom_hs = np.array(data['mom_hs'])
    mom_iq = np.array(data['mom_iq'])
    
    # Transformed data (centering on reference points)
    c2_mom_hs = mom_hs - 0.5
    c2_mom_iq = mom_iq - 100.0
    inter = c2_mom_hs * c2_mom_iq
    
    with pm.Model() as model:
        # Parameters
        beta = pm.Flat("beta", shape=4)
        sigma = pm.HalfFlat("sigma")
        
        # Linear predictor
        mu = beta[0] + beta[1] * c2_mom_hs + beta[2] * c2_mom_iq + beta[3] * inter
        
        # Likelihood
        kid_score_obs = pm.Normal("kid_score", mu=mu, sigma=sigma, observed=kid_score)

    return model