def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    M = data['M']
    T = data['T'] 
    y = data['y']  # shape (M, T)
    
    # Compute transformed data (same as Stan)
    s = np.sum(y, axis=1)  # row sums, shape (M,)
    C = np.sum(s > 0)  # number of observed individuals
    
    with pm.Model() as model:
        # Parameters with implicit uniform priors (bounded)
        omega = pm.Uniform("omega", lower=0, upper=1)
        mean_p = pm.Uniform("mean_p", lower=0, upper=1, shape=T)
        sigma = pm.Uniform("sigma", lower=0, upper=5)
        
        # Random effects (non-centered)
        eps_raw = pm.Normal("eps_raw", mu=0, sigma=1, shape=M)
        
        # Transformed parameters
        eps = pm.Deterministic("eps", sigma * eps_raw)
        mean_lp = pm.Deterministic("mean_lp", pm.math.logit(mean_p))
        
        # logit_p matrix: broadcast mean_lp[j] + eps[i] for all i,j
        logit_p = pm.Deterministic("logit_p", mean_lp[None, :] + eps[:, None])  # shape (M, T)
        
        # Convert data to tensors for easier manipulation
        y_tensor = pt.as_tensor_variable(y, dtype='int32')
        s_tensor = pt.as_tensor_variable(s, dtype='int32')
        
        # Likelihood computation
        # For each individual i, compute log probability
        def compute_individual_logp(i, logit_p_i, y_i, s_i, omega):
            # Bernoulli logp for all T occasions for individual i
            bernoulli_logp = pt.sum(pm.logp(pm.Bernoulli.dist(logit_p=logit_p_i), y_i))
            
            # If s[i] > 0: z[i] = 1 with probability 1
            detected_logp = pm.math.log(omega) + bernoulli_logp
            
            # If s[i] = 0: mixture of z[i] = 0 and z[i] = 1
            undetected_case1 = pm.math.log(omega) + bernoulli_logp  # z[i] = 1, not detected
            undetected_case0 = pm.math.log(1 - omega)  # z[i] = 0
            undetected_logp = pm.math.logsumexp(pt.stack([undetected_case1, undetected_case0]))
            
            # Use s[i] > 0 to select which case
            return pt.where(s_i > 0, detected_logp, undetected_logp)
        
        # Compute log probability for all individuals
        individual_logps = []
        for i in range(M):
            individual_logp = compute_individual_logp(
                i, logit_p[i], y_tensor[i], s_tensor[i], omega
            )
            individual_logps.append(individual_logp)
        
        total_logp = pt.sum(pt.stack(individual_logps))
        pm.Potential("likelihood", total_logp)

    return model