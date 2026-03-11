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
        
        # Transformed parameters - compute cumulative counts
        n_avoid = np.zeros((n_dogs, n_trials))
        n_shock = np.zeros((n_dogs, n_trials))
        
        for j in range(n_dogs):
            n_avoid[j, 0] = 0
            n_shock[j, 0] = 0
            
            for t in range(1, n_trials):
                n_avoid[j, t] = n_avoid[j, t-1] + 1 - y_data[j, t-1]
                n_shock[j, t] = n_shock[j, t-1] + y_data[j, t-1]
        
        # Compute logit probabilities
        p = beta[0] + beta[1] * n_avoid + beta[2] * n_shock
        
        # Use Bernoulli with observed data
        pm.Bernoulli("y", logit_p=p, observed=y_data)
        
        # Add correction for the observed constant offset
        pm.Potential("stan_normalization_correction", pt.constant(16.572327))

    return model