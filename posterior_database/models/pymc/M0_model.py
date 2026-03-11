def make_model(data: dict):
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    M = data['M']
    T = data['T'] 
    y = data['y']
    
    # Transformed data computations
    s = np.sum(y, axis=1)  # Totals in each row
    C = np.sum(s > 0)  # Size of observed data set
    
    with pm.Model() as model:
        # Parameters with implicit uniform priors
        omega = pm.Uniform("omega", lower=0, upper=1)  # Inclusion probability
        p = pm.Uniform("p", lower=0, upper=1)  # Detection probability
        
        # Likelihood - marginalized over latent z
        loglik_terms = []
        
        for i in range(M):
            if s[i] > 0:
                # z[i] == 1 case
                # bernoulli_lpmf(1 | omega) + binomial_lpmf(s[i] | T, p)
                loglik_i = (pt.log(omega) + 
                           pm.Binomial.logp(s[i], n=T, p=p))
            else:  # s[i] == 0
                # Log-sum-exp over z[i] == 1 and z[i] == 0
                # Case z[i] == 1: bernoulli_lpmf(1 | omega) + binomial_lpmf(0 | T, p)
                log_z1 = (pt.log(omega) + 
                         pm.Binomial.logp(0, n=T, p=p))
                # Case z[i] == 0: bernoulli_lpmf(0 | omega)
                log_z0 = pt.log(1 - omega)
                loglik_i = pm.math.logsumexp(pt.stack([log_z1, log_z0]))
            
            loglik_terms.append(loglik_i)
        
        # Add total likelihood
        pm.Potential("likelihood", pt.sum(pt.stack(loglik_terms)))
        
        # Generated quantities
        omega_nd = pm.Deterministic("omega_nd", 
                                   (omega * (1 - p)**T) / 
                                   (omega * (1 - p)**T + (1 - omega)))
    
    return model