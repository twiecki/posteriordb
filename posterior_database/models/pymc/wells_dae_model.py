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
    educ = np.array(data['educ'])
    
    # Transformed data - rescaling
    dist100 = dist / 100.0
    educ4 = educ / 4.0
    
    # Create design matrix [N, 3] with columns: dist100, arsenic, educ4
    x = np.column_stack([dist100, arsenic, educ4])

    with pm.Model() as model:
        # Parameters
        alpha = pm.Flat("alpha")
        beta = pm.Flat("beta", shape=3)
        
        # Linear predictor
        eta = alpha + x @ beta
        
        # Likelihood - bernoulli with logit link
        switched_obs = pm.Bernoulli("switched", logit_p=eta, observed=switched)

    return model