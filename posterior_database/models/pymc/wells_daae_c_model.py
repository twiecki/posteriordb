def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data and convert to numpy arrays
        N = data['N']
        switched = np.array(data['switched'])
        dist = np.array(data['dist'])
        arsenic = np.array(data['arsenic'])
        assoc = np.array(data['assoc'])
        educ = np.array(data['educ'])
        
        # Transformed data - replicate Stan's transformed data block
        c_dist100 = (dist - np.mean(dist)) / 100.0
        c_arsenic = arsenic - np.mean(arsenic)
        da_inter = c_dist100 * c_arsenic
        educ4 = educ / 4.0
        
        # Create design matrix x - shape [N, 5]
        x = np.column_stack([c_dist100, c_arsenic, da_inter, assoc, educ4])
        
        # Parameters - using Flat priors since Stan had no explicit priors
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", shape=5)
        
        # Model: bernoulli_logit_glm equivalent
        # switched ~ bernoulli_logit_glm(x, alpha, beta)
        # This is equivalent to: switched ~ bernoulli_logit(alpha + x * beta)
        logit_p = alpha + pm.math.dot(x, beta)
        
        switched_obs = pm.Bernoulli("switched", logit_p=logit_p, observed=switched)

    return model