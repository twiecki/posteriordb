def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        N = data['N']
        kid_score = data['kid_score']
        mom_hs = data['mom_hs']
        mom_iq = data['mom_iq']
        
        # Transformed data (centering predictors)
        c_mom_hs = mom_hs - np.mean(mom_hs)
        c_mom_iq = mom_iq - np.mean(mom_iq)
        inter = c_mom_hs * c_mom_iq
        
        # Parameters
        beta = pm.Flat("beta", shape=4)
        sigma = pm.HalfFlat("sigma")
        
        # Model
        mu = beta[0] + beta[1] * c_mom_hs + beta[2] * c_mom_iq + beta[3] * inter
        kid_score_obs = pm.Normal("kid_score", mu=mu, sigma=sigma, observed=kid_score)

    return model