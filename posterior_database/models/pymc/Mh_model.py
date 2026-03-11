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
        
        # Likelihood using custom potentials for each observation
        for i in range(M):
            if y[i] > 0:
                # Case: animal was detected at least once (z[i] == 1)
                logp_detected = (pm.math.log(omega) + 
                               pm.logp(pm.Binomial.dist(n=T, logit_p=eps[i]), y[i]))
                pm.Potential(f"likelihood_{i}", logp_detected)
            else:
                # Case: animal was never detected (y[i] == 0)
                # Two possibilities: present but not detected, or not present
                
                # z[i] == 1 (present but not detected)
                logp_present = (pm.math.log(omega) + 
                              pm.logp(pm.Binomial.dist(n=T, logit_p=eps[i]), 0))
                
                # z[i] == 0 (not present)
                logp_absent = pm.math.log(1 - omega)
                
                # Log-sum-exp of the two possibilities
                logp_total = pm.math.logaddexp(logp_present, logp_absent)
                pm.Potential(f"likelihood_{i}", logp_total)
    
    return model