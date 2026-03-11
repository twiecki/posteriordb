def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    n_dogs = data['n_dogs']
    n_trials = data['n_trials'] 
    y_data = np.array(data['y'])  # Ensure it's a numpy array
    
    J = n_dogs
    T = n_trials
    # Transformed data - compute prev_shock and prev_avoid (vectorized)
    prev_shock = np.hstack([np.zeros((J, 1)), np.cumsum(y_data[:, :-1], axis=1)])
    prev_avoid = np.hstack([np.zeros((J, 1)), np.cumsum(1 - y_data[:, :-1], axis=1)])

    with pm.Model() as model:
        # Parameters - both bounded to (0,1) with implicit uniform priors
        a = pm.Uniform("a", lower=0, upper=1)
        b = pm.Uniform("b", lower=0, upper=1)
        
        # Vectorized computation of probabilities
        p = a ** prev_shock * b ** prev_avoid
        
        # Create observed Bernoulli variables
        y_obs = pm.Bernoulli("y", p=p, observed=y_data)

    return model