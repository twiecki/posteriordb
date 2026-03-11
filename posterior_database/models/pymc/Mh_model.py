def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    M = data['M']
    T = data['T'] 
    y = data['y']
    
    # Convert y to numpy array if it isn't already
    y = np.array(y)
    
    # Compute transformed data (size of observed data set)
    C = np.sum(y > 0)
    
    with pm.Model() as model:
        # Parameters with implicit uniform priors (bounded parameters with no explicit priors)
        omega = pm.Uniform("omega", lower=0, upper=1)
        mean_p = pm.Uniform("mean_p", lower=0, upper=1) 
        sigma = pm.Uniform("sigma", lower=0, upper=5)
        
        # Raw effects for non-centered parameterization
        eps_raw = pm.Normal("eps_raw", mu=0, sigma=1, shape=M)
        
        # Transformed parameters
        eps = pm.Deterministic("eps", pm.math.logit(mean_p) + sigma * eps_raw)
        
        # Vectorized likelihood
        y_arr = np.array(y, dtype=np.float64)
        obs_mask = y > 0

        # Observed individuals: z[i] = 1
        logp_obs = (pm.math.log(omega) +
                    pm.logp(pm.Binomial.dist(n=T, logit_p=eps[obs_mask]), y_arr[obs_mask]))

        # Unobserved individuals: marginalize over z
        logp_present = (pm.math.log(omega) +
                        pm.logp(pm.Binomial.dist(n=T, logit_p=eps[~obs_mask]), 0))
        logp_absent = pm.math.log(1 - omega)
        logp_unobs = pm.math.logaddexp(logp_present, logp_absent)

        pm.Potential("likelihood", pt.sum(logp_obs) + pt.sum(logp_unobs))
    
    return model