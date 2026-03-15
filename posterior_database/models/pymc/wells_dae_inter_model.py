def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    N = data['N']
    switched = data['switched']
    dist = data['dist']
    arsenic = data['arsenic']
    educ = data['educ']
    c_dist100 = (dist - np.mean(dist)) / 100.0
    c_arsenic = arsenic - np.mean(arsenic)
    c_educ4 = (educ - np.mean(educ)) / 4.0
    da_inter = c_dist100 * c_arsenic
    de_inter = c_dist100 * c_educ4
    ae_inter = c_arsenic * c_educ4
    x = np.column_stack([c_dist100, c_arsenic, c_educ4, da_inter, de_inter, ae_inter])

    with pm.Model() as model:
        # Parameters - using flat priors (improper in Stan)
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", shape=6)
        
        # Linear predictor
        eta = alpha + x @ beta
        
        # Likelihood - bernoulli_logit_glm becomes Bernoulli with logit link
        switched_obs = pm.Bernoulli("switched", logit_p=eta, observed=switched)

    return model