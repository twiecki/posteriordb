def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    N = data['N']
    kid_score = data['kid_score']
    mom_hs = data['mom_hs']
    mom_iq = data['mom_iq']
    
    # Transformed data - standardizing (equivalent to Stan's transformed data block)
    z_mom_hs = (mom_hs - np.mean(mom_hs)) / (2 * np.std(mom_hs, ddof=0))
    z_mom_iq = (mom_iq - np.mean(mom_iq)) / (2 * np.std(mom_iq, ddof=0))
    inter = z_mom_hs * z_mom_iq

    with pm.Model() as model:
        # Parameters
        beta = pm.Flat("beta", shape=4)
        sigma = pm.HalfFlat("sigma")
        
        # Linear predictor
        mu = beta[0] + beta[1] * z_mom_hs + beta[2] * z_mom_iq + beta[3] * inter
        
        # Likelihood
        kid_score_obs = pm.Normal("kid_score", mu=mu, sigma=sigma, observed=kid_score)

    return model