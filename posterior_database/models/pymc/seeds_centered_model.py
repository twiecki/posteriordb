def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data and ensure proper numpy arrays
    I = data['I']
    n = np.array(data['n'])
    N = np.array(data['N'])
    x1 = np.array(data['x1'], dtype=float)
    x2 = np.array(data['x2'], dtype=float)
    
    # Transformed data
    x1x2 = x1 * x2
    
    with pm.Model() as model:
        # Parameters
        alpha0 = pm.Normal("alpha0", mu=0.0, sigma=1.0)
        alpha1 = pm.Normal("alpha1", mu=0.0, sigma=1.0)
        alpha12 = pm.Normal("alpha12", mu=0.0, sigma=1.0)
        alpha2 = pm.Normal("alpha2", mu=0.0, sigma=1.0)
        
        # Note: Stan's cauchy(0, 1) with lower=0 bound becomes HalfCauchy
        # But this creates a log(2) offset, so we need to correct for it
        sigma = pm.HalfCauchy("sigma", beta=1.0)
        
        # Random effects
        c = pm.Normal("c", mu=0.0, sigma=sigma, shape=I)
        
        # Transformed parameters - center the random effects
        b = pm.Deterministic("b", c - pt.mean(c))
        
        # Linear predictor
        eta = alpha0 + alpha1 * x1 + alpha2 * x2 + alpha12 * x1x2 + b
        
        # Likelihood - binomial with logit link
        n_obs = pm.Binomial("n", n=N, logit_p=eta, observed=n)
        
        # Correction for HalfCauchy log(2) offset to match Stan's unnormalized half-distribution
        pm.Potential("half_dist_correction", -pt.log(2.0))
    
    return model