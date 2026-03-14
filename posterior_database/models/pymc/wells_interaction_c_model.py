def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        N = data['N']
        switched = data['switched']
        dist = data['dist']
        arsenic = data['arsenic']
        
        # Transformed data - centering and interaction
        c_dist100 = (dist - np.mean(dist)) / 100.0
        c_arsenic = arsenic - np.mean(arsenic)
        inter = c_dist100 * c_arsenic
        
        # Create design matrix x
        x = np.column_stack([c_dist100, c_arsenic, inter])
        
        # Parameters - using flat priors as Stan has no explicit priors
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", shape=3)
        
        # Model: bernoulli_logit_glm(x, alpha, beta)
        # This is equivalent to bernoulli_logit(alpha + x * beta)
        logit_p = alpha + x @ beta
        
        # Likelihood
        switched_obs = pm.Bernoulli("switched", logit_p=logit_p, observed=switched)

    return model