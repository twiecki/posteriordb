def make_model(data: dict):
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    N = data['N']
    partyid7 = data['partyid7']
    real_ideo = data['real_ideo']
    race_adj = data['race_adj']
    educ1 = data['educ1']
    gender = data['gender']
    income = data['income']
    age_discrete = data['age_discrete']
    
    # Transformed data - create age group dummy variables
    age30_44 = np.zeros(N)
    age45_64 = np.zeros(N)
    age65up = np.zeros(N)
    
    for n in range(N):
        age30_44[n] = float(age_discrete[n] == 2)
        age45_64[n] = float(age_discrete[n] == 3)
        age65up[n] = float(age_discrete[n] == 4)
    
    with pm.Model() as model:
        # Parameters - using flat priors as in Stan (no explicit priors given)
        beta = pm.Flat("beta", shape=9)
        sigma = pm.HalfFlat("sigma")
        
        # Linear predictor
        mu = (beta[0] + 
              beta[1] * real_ideo + 
              beta[2] * race_adj +
              beta[3] * age30_44 + 
              beta[4] * age45_64 +
              beta[5] * age65up + 
              beta[6] * educ1 + 
              beta[7] * gender +
              beta[8] * income)
        
        # Likelihood
        y_obs = pm.Normal("partyid7", mu=mu, sigma=sigma, observed=partyid7)
        
    return model