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
        # Parameters
        alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1.0)
        alpha1 = pm.Normal("alpha1", mu=0.0, sigma=1.0)
        alpha2 = pm.Normal("alpha2", mu=0.0, sigma=1.0)
        alpha12 = pm.Normal("alpha12", mu=0.0, sigma=1.0)
        
        # sigma has lower bound 0 with Cauchy prior
        sigma = pm.HalfCauchy("sigma", beta=1.0)
        
        # Random effects
        b = pm.Normal("b", mu=0.0, sigma=sigma, shape=I)
        
        # Linear predictor (on logit scale)
        logit_p = alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2 + b
        
        # Likelihood: binomial with logit parameterization
        n_obs = pm.Binomial("n", n=N, logit_p=logit_p, observed=n)
        
    return model