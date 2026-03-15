def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    N = data['N']  # 1000 students
    R = data['R']  # 32 response patterns
    T = data['T']  # 5 questions
    culm = data['culm']  # cumulative counts
    response = data['response']  # [R, T] response patterns
    
    # Convert to numpy arrays
    culm = np.array(culm)
    response = np.array(response)
    
    # Transformed data: expand response patterns to individual students (vectorized)
    counts = np.diff(np.concatenate([[0], culm]))
    r = np.repeat(response, counts, axis=0).T  # shape (T, N)
    
    with pm.Model() as model:
        # Parameters
        alpha = pm.Normal("alpha", mu=0, sigma=100, shape=T)
        theta = pm.Normal("theta", mu=0, sigma=1, shape=N)
        
        # Stan: real<lower=0> beta ~ normal(0, 100) is HalfNormal in PyMC
        # But we need to account for the log(2) offset difference
        beta = pm.HalfNormal("beta", sigma=100)
        
        # Model: bernoulli_logit for all questions at once (vectorized)
        # Stan: r[k] ~ bernoulli_logit(beta * theta - alpha[k])
        logit_p = beta * theta[None, :] - alpha[:, None]  # shape (T, N)
        pm.Bernoulli("r", logit_p=logit_p, observed=r)
        
        # Generated quantities (as deterministic variables)
        mean_alpha = pm.Deterministic("mean_alpha", pt.mean(alpha))
        a = pm.Deterministic("a", alpha - mean_alpha)
    
    return model