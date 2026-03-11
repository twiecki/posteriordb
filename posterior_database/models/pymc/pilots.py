def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Convert 1-based indices to 0-based
    group_idx = np.array(data['group_id']) - 1
    scenario_idx = np.array(data['scenario_id']) - 1
    
    with pm.Model() as model:
        # Hyperparameters for group effects
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
        sigma_a = pm.Uniform("sigma_a", lower=0, upper=100)
        
        # Group effects
        a = pm.Normal("a", mu=10 * mu_a, sigma=sigma_a, shape=data['n_groups'])
        
        # Hyperparameters for scenario effects  
        mu_b = pm.Normal("mu_b", mu=0, sigma=1)
        sigma_b = pm.Uniform("sigma_b", lower=0, upper=100)
        
        # Scenario effects
        b = pm.Normal("b", mu=10 * mu_b, sigma=sigma_b, shape=data['n_scenarios'])
        
        # Observation noise
        sigma_y = pm.Uniform("sigma_y", lower=0, upper=100)
        
        # Linear combination of group and scenario effects
        y_hat = a[group_idx] + b[scenario_idx]
        
        # Likelihood
        y_obs = pm.Normal("y", mu=y_hat, sigma=sigma_y, observed=data['y'])
        
    return model