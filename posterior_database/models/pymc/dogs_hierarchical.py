def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    n_dogs = data['n_dogs']
    n_trials = data['n_trials'] 
    y_data = np.array(data['y'])  # Ensure it's a numpy array
    
    # Transformed data - compute prev_shock and prev_avoid
    J = n_dogs
    T = n_trials
    prev_shock = np.zeros((J, T))
    prev_avoid = np.zeros((J, T))
    
    # Build prev_shock and prev_avoid arrays
    for j in range(J):
        prev_shock[j, 0] = 0
        prev_avoid[j, 0] = 0
        for t in range(1, T):
            prev_shock[j, t] = prev_shock[j, t-1] + y_data[j, t-1]
            prev_avoid[j, t] = prev_avoid[j, t-1] + (1 - y_data[j, t-1])

    with pm.Model() as model:
        # Parameters - both bounded to (0,1) with implicit uniform priors
        a = pm.Uniform("a", lower=0, upper=1)
        b = pm.Uniform("b", lower=0, upper=1)
        
        # Vectorized computation of probabilities
        p = a ** prev_shock * b ** prev_avoid
        
        # Create observed Bernoulli variables
        y_obs = pm.Bernoulli("y", p=p, observed=y_data)

    return model