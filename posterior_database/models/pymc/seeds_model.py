def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data and ensure they are numpy arrays
    I = data['I']
    n = np.array(data['n'])
    N = np.array(data['N'])
    x1 = np.array(data['x1'])
    x2 = np.array(data['x2'])
    
    # Transformed data
    x1x2 = x1 * x2
    
    with pm.Model() as model:
        # Parameters
        alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1000.0)
        alpha1 = pm.Normal("alpha1", mu=0.0, sigma=1000.0)
        alpha2 = pm.Normal("alpha2", mu=0.0, sigma=1000.0)
        alpha12 = pm.Normal("alpha12", mu=0.0, sigma=1000.0)
        
        tau = pm.Gamma("tau", alpha=1e-3, beta=1e-3)
        
        # Transformed parameters
        sigma = pm.Deterministic("sigma", 1.0 / pt.sqrt(tau))
        
        # Random effects
        b = pm.Normal("b", mu=0.0, sigma=sigma, shape=I)
        
        # Linear predictor
        linear_pred = alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2 + b
        
        # Likelihood
        n_obs = pm.Binomial("n", n=N, logit_p=linear_pred, observed=n)

    return model