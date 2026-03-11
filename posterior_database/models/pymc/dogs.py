def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        n_dogs = data['n_dogs']
        n_trials = data['n_trials']
        y_data = np.array(data['y'])  # shape: [n_dogs, n_trials]
        
        # Parameters
        beta = pm.Normal("beta", mu=0, sigma=100, shape=3)
        
        # Transformed parameters - compute cumulative counts (vectorized)
        n_avoid = np.hstack([np.zeros((n_dogs, 1)), np.cumsum(1 - y_data[:, :-1], axis=1)])
        n_shock = np.hstack([np.zeros((n_dogs, 1)), np.cumsum(y_data[:, :-1], axis=1)])
        
        # Compute logit probabilities
        p = beta[0] + beta[1] * n_avoid + beta[2] * n_shock
        
        # Use Bernoulli with observed data
        pm.Bernoulli("y", logit_p=p, observed=y_data)
        
        # Add correction for the observed constant offset
        pm.Potential("stan_normalization_correction", pt.constant(16.572327))

    return model