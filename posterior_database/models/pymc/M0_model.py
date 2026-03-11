def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    y = data['y']  # shape [M, T]
    M = data['M']
    T = data['T']
    
    # Transformed data (compute row sums and observed count)
    s = np.sum(y, axis=1)  # sum across columns for each row
    C = np.sum(s > 0)  # count of individuals with at least one capture
    
    with pm.Model() as model:
        # Parameters with uniform priors (implicit in Stan)
        omega = pm.Uniform("omega", lower=0, upper=1)  # Inclusion probability
        p = pm.Uniform("p", lower=0, upper=1)  # Detection probability
        
        # Likelihood using custom potential
        # For each individual i, we have a mixture:
        # - If s[i] > 0: individual was observed, so z[i]=1 with prob omega, 
        #   and s[i] captures ~ binomial(T, p)
        # - If s[i] == 0: either z[i]=1 with no captures, or z[i]=0
        
        logp_contributions = []
        
        for i in range(M):
            if s[i] > 0:
                # Individual was observed: z[i] = 1
                # log P(z[i]=1) + log P(s[i] | z[i]=1, T, p)
                # Use the binomial log probability manually
                log_binom_coeff = pt.gammaln(T + 1) - pt.gammaln(s[i] + 1) - pt.gammaln(T - s[i] + 1)
                contrib = (pt.log(omega) + 
                          log_binom_coeff + s[i] * pt.log(p) + (T - s[i]) * pt.log(1 - p))
            else:
                # Individual not observed: could be z[i]=1 with no captures or z[i]=0
                # log P(s[i]=0) = log[P(z[i]=1) * P(s[i]=0|z[i]=1) + P(z[i]=0)]
                log_prob_present_not_detected = pt.log(omega) + T * pt.log(1 - p)
                log_prob_absent = pt.log(1 - omega)
                contrib = pm.math.logsumexp(pt.stack([log_prob_present_not_detected, log_prob_absent]))
            
            logp_contributions.append(contrib)
        
        # Add all contributions to the model
        pm.Potential("likelihood", pt.sum(pt.stack(logp_contributions)))
        
        # Generated quantities
        omega_nd = pm.Deterministic("omega_nd", 
                                   (omega * (1 - p)**T) / (omega * (1 - p)**T + (1 - omega)))
        
    return model