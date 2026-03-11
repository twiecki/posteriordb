def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data and ensure numpy arrays
    n = data['n']
    C = np.array(data['C'])
    year = np.array(data['year'])
    
    # Transformed data (compute before model)
    year_squared = year * year
    year_cubed = year * year * year
    
    with pm.Model() as model:
        # Parameters with uniform priors (matching Stan bounds)
        alpha = pm.Uniform("alpha", lower=-20, upper=20)
        beta1 = pm.Uniform("beta1", lower=-10, upper=10)
        beta2 = pm.Uniform("beta2", lower=-10, upper=20)
        beta3 = pm.Uniform("beta3", lower=-10, upper=10)
        sigma = pm.Uniform("sigma", lower=0, upper=5)
        
        # Random year effects
        eps = pm.Normal("eps", mu=0, sigma=sigma, shape=n)
        
        # Transformed parameters: log_lambda
        log_lambda = pm.Deterministic("log_lambda", 
            alpha + beta1 * year + beta2 * year_squared + beta3 * year_cubed + eps)
        
        # Likelihood
        C_obs = pm.Poisson("C", mu=pt.exp(log_lambda), observed=C)
        
        # Generated quantities
        lambda_param = pm.Deterministic("lambda", pt.exp(log_lambda))
    
    return model