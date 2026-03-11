def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    y = data['y']  # shape (M, T)
    M = data['M']
    T = data['T']
    
    # Transformed data - compute totals for each individual
    s = np.sum(y, axis=1)  # sum across time periods for each individual
    C = np.sum(s > 0)  # number of individuals with at least one capture

    with pm.Model() as model:
        # Parameters with implicit uniform priors
        omega = pm.Uniform("omega", lower=0, upper=1)  # inclusion probability
        p = pm.Uniform("p", lower=0, upper=1, shape=T)  # detection probabilities
        
        # Likelihood
        # For individuals with s[i] > 0 (observed at least once)
        observed_mask = s > 0
        observed_indices = np.where(observed_mask)[0]
        
        # For individuals with s[i] == 0 (never observed)
        unobserved_mask = s == 0
        unobserved_indices = np.where(unobserved_mask)[0]
        
        # Likelihood for observed individuals: z[i] = 1 (definitely present)
        for i in observed_indices:
            # bernoulli_lpmf(1 | omega) - this is just log(omega)
            pm.Potential(f"omega_contrib_{i}", pt.log(omega))
            
            # bernoulli_lpmf(y[i] | p) - vectorized Bernoulli likelihood
            pm.Bernoulli(f"y_obs_{i}", p=p, observed=y[i])
        
        # Likelihood for unobserved individuals: marginalize over z[i]
        for i in unobserved_indices:
            # log_sum_exp of two terms:
            # Term 1: z[i] = 1, bernoulli_lpmf(1|omega) + bernoulli_lpmf(y[i]|p)
            # Term 2: z[i] = 0, bernoulli_lpmf(0|omega)
            
            # y[i] is all zeros for unobserved individuals
            # bernoulli_lpmf(y[i]|p) = sum(log(1-p)) when y[i] are all zeros
            log_prob_y_given_present = pt.sum(pt.log(1 - p))
            
            term1 = pt.log(omega) + log_prob_y_given_present  # z[i] = 1
            term2 = pt.log(1 - omega)  # z[i] = 0
            
            pm.Potential(f"unobs_contrib_{i}", pm.math.logsumexp(pt.stack([term1, term2])))
        
        # Generated quantities (as deterministics)
        pr = pm.Deterministic("pr", pt.prod(1 - p))  # prob never captured given present
        omega_nd = pm.Deterministic("omega_nd", (omega * pr) / (omega * pr + (1 - omega)))
        
    return model