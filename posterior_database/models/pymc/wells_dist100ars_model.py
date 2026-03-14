def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        N = data['N']
        switched = data['switched']
        dist = np.array(data['dist'])
        arsenic = np.array(data['arsenic'])
        
        # Transformed data - rescaling
        dist100 = dist / 100.0
        
        # Create design matrix [dist100, arsenic] - shape (N, 2)
        x = np.column_stack([dist100, arsenic])
        
        # Parameters
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", shape=2)
        
        # Model - Bernoulli logistic GLM
        # Linear predictor: alpha + x @ beta
        eta = alpha + x @ beta
        
        # Likelihood
        switched_obs = pm.Bernoulli("switched", logit_p=eta, observed=switched)

    return model