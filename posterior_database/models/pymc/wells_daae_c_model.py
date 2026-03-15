def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    N = data['N']
    switched = np.array(data['switched'])
    dist = np.array(data['dist'])
    arsenic = np.array(data['arsenic'])
    assoc = np.array(data['assoc'])
    educ = np.array(data['educ'])
    c_dist100 = (dist - np.mean(dist)) / 100.0
    c_arsenic = arsenic - np.mean(arsenic)
    da_inter = c_dist100 * c_arsenic
    educ4 = educ / 4.0
    x = np.column_stack([c_dist100, c_arsenic, da_inter, assoc, educ4])

    with pm.Model() as model:
        # Parameters - using Flat priors since Stan had no explicit priors
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", shape=5)
        
        # Model: bernoulli_logit_glm equivalent
        # switched ~ bernoulli_logit_glm(x, alpha, beta)
        # This is equivalent to: switched ~ bernoulli_logit(alpha + x * beta)
        logit_p = alpha + x @ beta
        
        switched_obs = pm.Bernoulli("switched", logit_p=logit_p, observed=switched)

    return model