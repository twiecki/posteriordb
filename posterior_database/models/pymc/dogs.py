def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data and ensure numpy arrays
        n_dogs = data['n_dogs']
        n_trials = data['n_trials']
        y = np.array(data['y'])
        
        # Parameters
        beta = pm.Normal("beta", mu=0, sigma=100, shape=3)
        
        # Transformed parameters - compute cumulative counts
        # We need to compute cumulative sums of (1-y) and y, but with a lag
        # n_avoid[j,t] = sum of (1 - y[j, 0:t-1]) for t >= 1, 0 for t=0
        # n_shock[j,t] = sum of y[j, 0:t-1] for t >= 1, 0 for t=0
        
        # Create shifted cumulative sums
        # For avoid: cumsum of (1-y), but shifted right by 1 position (prepend 0)
        avoid_cumsum = pt.cumsum(1 - y, axis=1)  # cumsum along trials
        n_avoid = pt.concatenate([pt.zeros((n_dogs, 1)), avoid_cumsum[:, :-1]], axis=1)
        
        # For shock: cumsum of y, but shifted right by 1 position (prepend 0)  
        shock_cumsum = pt.cumsum(y, axis=1)  # cumsum along trials
        n_shock = pt.concatenate([pt.zeros((n_dogs, 1)), shock_cumsum[:, :-1]], axis=1)
        
        # Compute probabilities (logit scale)
        p = beta[0] + beta[1] * n_avoid + beta[2] * n_shock
        
        # Likelihood
        y_obs = pm.Bernoulli("y", logit_p=p, observed=y)

    return model