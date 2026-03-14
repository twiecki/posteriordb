def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data and convert to numpy arrays
    I = data['I']
    n = np.array(data['n'])
    N = np.array(data['N']) 
    x1 = np.array(data['x1'])
    x2 = np.array(data['x2'])
    
    # Transformed data: interaction term
    x1x2 = x1 * x2
    
    with pm.Model() as model:
        # Fixed effect parameters
        alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1.0)
        alpha1 = pm.Normal("alpha1", mu=0.0, sigma=1.0)
        alpha2 = pm.Normal("alpha2", mu=0.0, sigma=1.0)
        alpha12 = pm.Normal("alpha12", mu=0.0, sigma=1.0)
        
        # Scale parameter for random effects
        # Stan: sigma ~ cauchy(0, 1) with <lower=0> constraint
        # This needs correction for log(2) offset
        sigma = pm.HalfCauchy("sigma", beta=1.0)
        
        # Random effects
        b = pm.Normal("b", mu=0.0, sigma=sigma, shape=I)
        
        # Linear predictor
        logit_p = alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2 + b
        
        # Likelihood
        n_obs = pm.Binomial("n", n=N, logit_p=logit_p, observed=n)
        
    
    return model