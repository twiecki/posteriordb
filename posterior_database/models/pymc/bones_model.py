def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    with pm.Model() as model:
        # Extract data as numpy arrays
        nChild = int(data['nChild'])
        nInd = int(data['nInd']) 
        gamma = np.array(data['gamma'])  # shape: [nInd, 4]
        delta = np.array(data['delta'])  # shape: [nInd]
        ncat = np.array(data['ncat'])    # shape: [nInd]
        grade = np.array(data['grade'])  # shape: [nChild, nInd]
        
        # Parameters - use Flat priors and add manual normal log prob to match Stan exactly
        theta = pm.Flat("theta", shape=nChild)
        
        # Add manual prior to match Stan's normal(0, 36) exactly
        # Stan might use unnormalized density (proportional to exp(-0.5 * (theta/36)^2))
        prior_log_prob = -0.5 * pt.sum((theta / 36.0) ** 2)
        pm.Potential("theta_prior", prior_log_prob)
        
        # Build the likelihood exactly as in Stan
        total_log_prob = 0.0
        
        for i in range(nChild):
            for j in range(nInd):
                # Check if this observation is not missing
                if grade[i, j] != -1:
                    n_cat = int(ncat[j])
                    
                    # Compute cumulative probabilities Q
                    Q_list = []
                    for k in range(n_cat - 1):  # k: 0 to n_cat-2 (0-based)
                        Q_k = pm.math.invlogit(delta[j] * (theta[i] - gamma[j, k]))
                        Q_list.append(Q_k)
                    
                    # Compute category probabilities p
                    p_list = []
                    
                    if n_cat == 1:
                        p_list = [1.0]
                    elif n_cat == 2:
                        p_list = [1 - Q_list[0], Q_list[0]]
                    else:
                        # First category
                        p_list.append(1 - Q_list[0])
                        
                        # Middle categories
                        for k in range(1, n_cat - 1):
                            p_list.append(Q_list[k-1] - Q_list[k])
                        
                        # Last category
                        p_list.append(Q_list[n_cat - 2])
                    
                    # Add log probability
                    obs_grade = int(grade[i, j])
                    grade_idx = obs_grade - 1  # Convert to 0-based
                    
                    if 0 <= grade_idx < len(p_list):
                        log_p = pt.log(p_list[grade_idx])
                        total_log_prob = total_log_prob + log_p
        
        # Add likelihood potential
        pm.Potential("likelihood", total_log_prob)
        
    return model