def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data and convert to numpy arrays
    N = data['N']
    switched = np.array(data['switched'])
    dist = np.array(data['dist'])
    arsenic = np.array(data['arsenic'])
    
    # Transformed data (rescaling and interaction)
    dist100 = dist / 100.0
    inter = dist100 * arsenic
    
    # Create design matrix [dist100, arsenic, inter]
    X = np.column_stack([dist100, arsenic, inter])

    with pm.Model() as model:
        # Parameters - using flat priors to match Stan's implicit priors
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", shape=3)
        
        # Linear predictor
        logit_p = alpha + X @ beta
        
        # Likelihood
        switched_obs = pm.Bernoulli("switched", logit_p=logit_p, observed=switched)

    return model