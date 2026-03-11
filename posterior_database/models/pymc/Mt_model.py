def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    y = np.array(data['y'])  # shape [M, T]
    M = data['M']
    T = data['T']
    
    # Transformed data - compute row sums and count observed individuals
    s = np.sum(y, axis=1)  # row sums
    C = np.sum(s > 0)  # count of observed individuals
    
    with pm.Model() as model:
        # Parameters with implicit uniform priors
        omega = pm.Uniform("omega", lower=0, upper=1)
        p = pm.Uniform("p", lower=0, upper=1, shape=T)
        
        # Likelihood
        # We'll handle each individual separately using pm.Potential
        for i in range(M):
            y_i = y[i]  # Get the capture history for individual i as numpy array
            if s[i] > 0:
                # Individual was captured at least once: z[i] = 1
                # Log probability: log(omega) + sum(log(Bernoulli(y[i,t] | p[t])))
                log_prob_zi_1 = pt.log(omega) + pt.sum(y_i * pt.log(p) + (1 - y_i) * pt.log(1 - p))
                pm.Potential(f"likelihood_{i}", log_prob_zi_1)
            else:
                # Individual was never captured: mixture of z[i] = 0 and z[i] = 1
                # z[i] = 1: log(omega) + sum(log(1-p)) (never detected but present)
                log_prob_zi_1 = pt.log(omega) + pt.sum(pt.log(1 - p))
                # z[i] = 0: log(1-omega) (not present)
                log_prob_zi_0 = pt.log(1 - omega)
                # Log-sum-exp of the two possibilities
                log_prob = pt.logsumexp(pt.stack([log_prob_zi_1, log_prob_zi_0]))
                pm.Potential(f"likelihood_{i}", log_prob)
        
        # Generated quantities as deterministic variables
        pr = pm.Deterministic("pr", pt.prod(1 - p))
        omega_nd = pm.Deterministic("omega_nd", (omega * pr) / (omega * pr + (1 - omega)))
        
    return model