def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data and convert to numpy arrays
    y = np.array(data['y'])
    M = data['M']
    T = data['T']
    
    # Transformed data: count observed animals (C)
    C = np.sum(y > 0)

    with pm.Model() as model:
        # Parameters with implicit uniform priors (bounded with no explicit prior)
        omega = pm.Uniform("omega", lower=0, upper=1)  # Inclusion probability
        mean_p = pm.Uniform("mean_p", lower=0, upper=1)  # Mean detection probability
        sigma = pm.Uniform("sigma", lower=0, upper=5)
        
        # Non-centered parameterization
        eps_raw = pm.Normal("eps_raw", mu=0, sigma=1, shape=M)
        
        # Transformed parameters
        eps = pm.Deterministic("eps", pm.math.logit(mean_p) + sigma * eps_raw)
        
        # Custom likelihood using a function approach
        def custom_logp(omega, eps, y):
            # Bernoulli log probability for being in the population
            log_omega = pt.log(omega)
            log_one_minus_omega = pt.log(1 - omega)
            
            # Detection probabilities on logit scale
            p_logit = eps
            
            # For observed animals (y > 0): must be present
            observed = pt.gt(y, 0)
            
            # Binomial logit log probability
            # binomial_logit_lpmf(k | n, alpha) = k * alpha - n * log1p_exp(alpha) + log_choose(n, k)
            # For now, ignore the binomial coefficient as it's constant for fixed n and doesn't affect sampling
            binomial_logp = (y * p_logit - T * pt.log1p(pt.exp(p_logit)))
            
            # For observed animals: log(omega) + binomial_logit_lpmf(y[i] | T, eps[i])
            obs_contribution = observed * (log_omega + binomial_logp)
            
            # For unobserved animals (y = 0): marginalize
            unobserved = pt.eq(y, 0)
            
            # Case 1: present but never detected (y = 0)
            binomial_0_logp = -T * pt.log1p(pt.exp(p_logit))  # 0 * p_logit = 0
            present_never_detected = log_omega + binomial_0_logp
            
            # Case 2: not present (broadcast scalar to vector)
            not_present = pt.full_like(present_never_detected, log_one_minus_omega)
            
            # Log-sum-exp for marginalization
            # Stack along new dimension and then logsumexp
            logp_stack = pt.stack([present_never_detected, not_present])
            unobs_logp = pm.math.logsumexp(logp_stack, axis=0)
            unobs_contribution = unobserved * unobs_logp
            
            return pt.sum(obs_contribution + unobs_contribution)
        
        pm.Potential("likelihood", custom_logp(omega, eps, y))

    return model