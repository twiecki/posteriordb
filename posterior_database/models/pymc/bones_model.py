def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        nChild = data['nChild']
        nInd = data['nInd']
        gamma = data['gamma']  # shape [nInd, 4]
        delta = data['delta']  # shape [nInd]
        ncat = data['ncat']  # shape [nInd]
        grade = data['grade']  # shape [nChild, nInd]
        
        # Parameters
        theta = pm.Normal("theta", mu=0.0, sigma=6.0, shape=nChild)
        
        # Custom log probability for the categorical likelihood with missing data
        def logp_func(theta):
            total_logp = 0.0
            
            for i in range(nChild):
                for j in range(nInd):
                    if grade[i][j] != -1:  # Only process non-missing observations
                        # Cumulative probabilities Q[i, j, k] for k = 1 to ncat[j]-1
                        Q_list = []
                        for k in range(ncat[j] - 1):
                            Q_k = pm.math.invlogit(delta[j] * (theta[i] - gamma[j][k]))
                            Q_list.append(Q_k)
                        
                        # Category probabilities p[i, j, k] for k = 1 to ncat[j]
                        p_list = []
                        
                        # p[1] = 1 - Q[1] (Stan uses 1-based indexing)
                        p_1 = 1 - Q_list[0]
                        p_list.append(p_1)
                        
                        # p[k] = Q[k-1] - Q[k] for k = 2 to ncat[j]-1
                        for k in range(1, ncat[j] - 1):
                            p_k = Q_list[k-1] - Q_list[k]
                            p_list.append(p_k)
                        
                        # p[ncat[j]] = Q[ncat[j]-1] (last category)
                        p_last = Q_list[ncat[j] - 2]
                        p_list.append(p_last)
                        
                        # Add log probability for observed grade (convert from 1-based to 0-based)
                        grade_idx = grade[i][j] - 1
                        total_logp += pt.log(p_list[grade_idx])
            
            return total_logp
        
        # Add the custom likelihood using Potential
        pm.Potential("grade_likelihood", logp_func(theta))

    return model