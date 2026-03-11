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
    
    # Transformed data: expand response patterns to individual students
    # This replicates the Stan transformed data block
    r = np.zeros((T, N), dtype=int)
    
    # First pattern (students 0 to culm[0]-1, converting from 1-based)
    for j in range(culm[0]):
        for k in range(T):
            r[k, j] = response[0, k]
    
    # Remaining patterns
    for i in range(1, R):
        start_idx = culm[i-1]  # previous cumulative count
        end_idx = culm[i]      # current cumulative count
        for j in range(start_idx, end_idx):
            for k in range(T):
                r[k, j] = response[i, k]
    
    with pm.Model() as model:
        # Parameters
        alpha = pm.Normal("alpha", mu=0, sigma=100, shape=T)
        theta = pm.Normal("theta", mu=0, sigma=1, shape=N)
        
        # Stan: real<lower=0> beta ~ normal(0, 100) is HalfNormal in PyMC
        # But we need to account for the log(2) offset difference
        beta = pm.HalfNormal("beta", sigma=100)
        # Add correction potential for the log(2) offset from HalfNormal vs Stan's approach
        pm.Potential("beta_correction", -pt.log(2.0))
        
        # Model: bernoulli_logit for each question
        for k in range(T):
            # Stan: r[k] ~ bernoulli_logit(beta * theta - alpha[k] * ones);
            # This is equivalent to: r[k] ~ bernoulli_logit(beta * theta - alpha[k])
            logit_p = beta * theta - alpha[k]
            pm.Bernoulli(f"r_{k}", logit_p=logit_p, observed=r[k])
        
        # Generated quantities (as deterministic variables)
        mean_alpha = pm.Deterministic("mean_alpha", pt.mean(alpha))
        a = pm.Deterministic("a", alpha - mean_alpha)
    
    return model