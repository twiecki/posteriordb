def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    N = data['N']  # 1000 students
    R = data['R']  # 32 response patterns  
    T = data['T']  # 5 questions
    culm = np.array(data['culm'])  # cumulative counts
    response = np.array(data['response'])  # response patterns [R, T]
    
    # Transformed data: expand patterns to individual responses
    # r[k,j] gives the response of student j to question k
    r = np.zeros((T, N), dtype=int)
    
    # First pattern (Stan uses 1-based indexing)
    for j in range(culm[0]):  # j from 0 to culm[0]-1 (Python 0-based)
        for k in range(T):
            r[k, j] = response[0, k]  # response pattern 0 (first pattern)
    
    # Remaining patterns  
    for i in range(1, R):  # i from 1 to R-1
        start_idx = culm[i-1]   # culm[i-1] + 1 in Stan, but 0-based here 
        end_idx = culm[i]       # culm[i] in Stan
        for j in range(start_idx, end_idx):
            for k in range(T):
                r[k, j] = response[i, k]  # response pattern i
    
    # Create ones vector
    ones = np.ones(N)
    
    with pm.Model() as model:
        # Parameters  
        alpha = pm.Normal("alpha", mu=0, sigma=100, shape=T)
        theta = pm.Normal("theta", mu=0, sigma=1, shape=N) 
        beta = pm.HalfNormal("beta", sigma=100)  # real<lower=0> beta constraint
        
        # Model: for each question k, model all student responses
        for k in range(T):
            # Linear predictor: beta * theta - alpha[k] * ones
            logit_p = beta * theta - alpha[k] * ones
            
            # Bernoulli logit likelihood
            pm.Bernoulli(f"r_{k}", logit_p=logit_p, observed=r[k, :])
        
        # Generated quantities
        mean_alpha = pm.Deterministic("mean_alpha", pt.mean(alpha))
        a = pm.Deterministic("a", alpha - mean_alpha)
    
    return model