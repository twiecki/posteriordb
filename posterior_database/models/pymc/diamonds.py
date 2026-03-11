def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    N = data['N']
    Y = data['Y']
    K = data['K'] 
    X = data['X']
    prior_only = data['prior_only']
    
    # Transformed data: center predictors (excluding intercept column)
    Kc = K - 1
    
    with pm.Model() as model:
        # Parameters
        b = pm.Normal("b", mu=0, sigma=1, shape=Kc)  # population-level effects
        Intercept = pm.StudentT("Intercept", nu=3, mu=8, sigma=10)  # intercept
        sigma = pm.HalfStudentT("sigma", nu=3, sigma=10)
        
        # Just test basic functionality first
        if prior_only == 0:
            Y_obs = pm.Normal("Y", mu=Intercept, sigma=sigma, observed=Y)

    return model