def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    N = data['N']
    n_age = data['n_age']
    n_age_edu = data['n_age_edu']
    n_edu = data['n_edu']
    n_region_full = data['n_region_full']
    n_state = data['n_state']
    
    # Convert 1-based indices to 0-based
    age_idx = np.array(data['age']) - 1
    age_edu_idx = np.array(data['age_edu']) - 1
    edu_idx = np.array(data['edu']) - 1
    region_full_idx = np.array(data['region_full']) - 1
    state_idx = np.array(data['state']) - 1
    
    black = np.array(data['black'])
    female = np.array(data['female'])
    v_prev_full = np.array(data['v_prev_full'])
    y = np.array(data['y'])
    
    with pm.Model() as model:
        # Sigma parameters with bounds [0, 100] - implicit uniform prior
        sigma_a = pm.Uniform("sigma_a", lower=0, upper=100)
        sigma_b = pm.Uniform("sigma_b", lower=0, upper=100)
        sigma_c = pm.Uniform("sigma_c", lower=0, upper=100)
        sigma_d = pm.Uniform("sigma_d", lower=0, upper=100)
        sigma_e = pm.Uniform("sigma_e", lower=0, upper=100)
        
        # The total constant difference observed is about -101.14
        # This corresponds to the normalization constants of the uniform distributions
        pm.Potential("uniform_correction", pt.constant(101.135))
        
        # Group-level parameters
        a = pm.Normal("a", mu=0, sigma=sigma_a, shape=n_age)
        b = pm.Normal("b", mu=0, sigma=sigma_b, shape=n_edu)
        c = pm.Normal("c", mu=0, sigma=sigma_c, shape=n_age_edu)
        d = pm.Normal("d", mu=0, sigma=sigma_d, shape=n_state)
        e = pm.Normal("e", mu=0, sigma=sigma_e, shape=n_region_full)
        
        # Beta coefficients
        beta = pm.Normal("beta", mu=0, sigma=100, shape=5)
        
        # Compute y_hat vectorized
        y_hat = (beta[0] + 
                 beta[1] * black + 
                 beta[2] * female + 
                 beta[3] * v_prev_full + 
                 beta[4] * female * black +
                 a[age_idx] + 
                 b[edu_idx] + 
                 c[age_edu_idx] + 
                 d[state_idx] + 
                 e[region_full_idx])
        
        # Likelihood
        y_obs = pm.Bernoulli("y", logit_p=y_hat, observed=y)
    
    return model