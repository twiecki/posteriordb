def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data and ensure numpy arrays
    I = data['I']
    n = np.array(data['n'])
    N = np.array(data['N']) 
    x1 = np.array(data['x1'])
    x2 = np.array(data['x2'])
    
    # Transformed data: element-wise multiplication
    x1x2 = x1 * x2
    
    with pm.Model() as model:
        # Parameters with priors
        alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1.0)
        alpha1 = pm.Normal("alpha1", mu=0.0, sigma=1.0)
        alpha12 = pm.Normal("alpha12", mu=0.0, sigma=1.0)
        alpha2 = pm.Normal("alpha2", mu=0.0, sigma=1.0)
        
        # Half-Cauchy for positive sigma
        sigma = pm.HalfCauchy("sigma", beta=1.0)
        
        # Random effects c ~ normal(0, sigma)
        c = pm.Normal("c", mu=0.0, sigma=sigma, shape=I)
        
        # Transformed parameter: b = c - mean(c)
        b = pm.Deterministic("b", c - pt.mean(c))
        
        # Linear predictor
        linear_pred = alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2 + b
        
        # Likelihood: binomial with logit link
        n_obs = pm.Binomial("n", n=N, logit_p=linear_pred, observed=n)
    
    return model