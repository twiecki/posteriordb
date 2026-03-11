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
        
        # Vectorized likelihood
        y_arr = np.array(y, dtype=np.float64)
        obs_mask = s > 0

        # Bernoulli logit logp for all individuals: shape (M, T)
        bern_logp = y_arr * logit_p - pm.math.log1pexp(logit_p)
        bern_logp_sum = pt.sum(bern_logp, axis=1)  # shape (M,)

        # Observed individuals: z[i] = 1
        logp_obs = pm.math.log(omega) + bern_logp_sum[obs_mask]

        # Unobserved individuals: marginalize over z
        logp_z1 = pm.math.log(omega) + bern_logp_sum[~obs_mask]
        logp_z0 = pm.math.log(1 - omega)
        logp_unobs = pm.math.logaddexp(logp_z1, logp_z0)

        pm.Potential("likelihood", pt.sum(logp_obs) + pt.sum(logp_unobs))
        
        # Generated quantities (for completeness, though not needed for logp)
        p = pm.Deterministic("p", pm.math.invlogit(logit_p))

    return model