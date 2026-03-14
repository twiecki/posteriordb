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
        educ = np.array(data['educ'])
        
        # Transformed data (following Stan code exactly)
        c_dist100 = (dist - np.mean(dist)) / 100.0
        c_arsenic = arsenic - np.mean(arsenic)
        da_inter = c_dist100 * c_arsenic
        educ4 = educ / 4.0
        
        # Create design matrix x (N x 4)
        x = np.column_stack([c_dist100, c_arsenic, da_inter, educ4])
        
        # Parameters (using flat priors to match Stan's implicit priors)
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", shape=4)
        
        # Linear predictor
        eta = alpha + x @ beta
        
        # Likelihood (bernoulli_logit_glm equivalent)
        y_obs = pm.Bernoulli("switched", logit_p=eta, observed=switched)

    return model