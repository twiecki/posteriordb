def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data and ensure they are numpy arrays
    n = data['n']
    C = np.asarray(data['C'])
    year = np.asarray(data['year'])
    
    # Transformed data (computed before model)
    year_squared = year**2
    year_cubed = year_squared * year

    with pm.Model() as model:
        # Parameters with uniform priors (bounded constraints without explicit priors)
        alpha = pm.Uniform("alpha", lower=-20, upper=20)
        beta1 = pm.Uniform("beta1", lower=-10, upper=10)
        beta2 = pm.Uniform("beta2", lower=-10, upper=10)
        beta3 = pm.Uniform("beta3", lower=-10, upper=10)
        
        # Transformed parameters
        log_lambda = pm.Deterministic("log_lambda", 
            alpha + beta1 * year + beta2 * year_squared + beta3 * year_cubed)
        
        # Likelihood - Poisson with log parameterization
        C_obs = pm.Poisson("C", mu=pt.exp(log_lambda), observed=C)
        
        # Generated quantities
        lambda_gq = pm.Deterministic("lambda", pt.exp(log_lambda))

    return model