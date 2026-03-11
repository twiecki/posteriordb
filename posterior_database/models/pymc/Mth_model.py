def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    M = data['M']
    T = data['T']
    y = data['y']  # shape [M, T]
    
    # Compute transformed data (equivalent to Stan's transformed data block)
    s = np.sum(y, axis=1)  # row sums
    C = np.sum(s > 0)  # number of animals detected at least once
    
    with pm.Model() as model:
        # Parameters
        # Stan: real<lower=0, upper=1> omega; (implicit uniform prior)
        omega = pm.Uniform("omega", lower=0, upper=1)
        
        # Stan: array[T] real<lower=0, upper=1> mean_p; (implicit uniform prior)
        mean_p = pm.Uniform("mean_p", lower=0, upper=1, shape=T)
        
        # Stan: real<lower=0, upper=5> sigma; (implicit uniform prior)
        sigma = pm.Uniform("sigma", lower=0, upper=5)
        
        # Stan: vector[M] eps_raw ~ normal(0, 1);
        eps_raw = pm.Normal("eps_raw", mu=0, sigma=1, shape=M)
        
        # Transformed parameters
        eps = sigma * eps_raw  # Random effects
        mean_lp = pm.math.logit(mean_p)  # logit of mean detection probabilities
        
        # Create logit_p matrix: logit_p[i,j] = mean_lp[j] + eps[i]
        # Broadcasting: mean_lp has shape (T,), eps has shape (M,)
        # We want result with shape (M, T)
        logit_p = mean_lp[None, :] + eps[:, None]  # shape (M, T)
        
        # Custom likelihood using pm.Potential
        # For each individual i:
        for i in range(M):
            if s[i] > 0:
                # Animal was detected at least once: z[i] = 1
                # target += bernoulli_lpmf(1 | omega) + bernoulli_logit_lpmf(y[i] | logit_p[i])
                log_prob_z1 = pm.math.log(omega)
                log_prob_y_given_z1 = pm.math.sum(
                    y[i] * logit_p[i] - pm.math.log1pexp(logit_p[i])
                )
                pm.Potential(f"likelihood_{i}", log_prob_z1 + log_prob_y_given_z1)
            else:
                # Animal was never detected: marginalize over z[i]
                # target += log_sum_exp(bernoulli_lpmf(1|omega) + bernoulli_logit_lpmf(y[i]|logit_p[i]),
                #                       bernoulli_lpmf(0|omega))
                
                # z[i] = 1 case
                log_prob_z1 = pm.math.log(omega)
                log_prob_y_given_z1 = pm.math.sum(
                    y[i] * logit_p[i] - pm.math.log1pexp(logit_p[i])
                )
                log_joint_z1 = log_prob_z1 + log_prob_y_given_z1
                
                # z[i] = 0 case  
                log_prob_z0 = pm.math.log(1 - omega)
                
                # log_sum_exp of the two cases
                log_marginal = pm.math.logsumexp(pt.stack([log_joint_z1, log_prob_z0]))
                pm.Potential(f"likelihood_{i}", log_marginal)
        
        # Generated quantities (for completeness, though not needed for logp)
        p = pm.Deterministic("p", pm.math.invlogit(logit_p))

    return model