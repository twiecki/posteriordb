def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    y = data['y']  # [M, T] capture history matrix
    M = data['M']  # Number of individuals in augmented dataset
    T = data['T']  # Number of sampling occasions

    # Transformed data - compute row sums and observed count
    s = np.sum(y, axis=1)  # Row sums
    C = np.sum(s > 0)  # Number of individuals ever captured

    with pm.Model() as model:
        # Parameters (uniform priors on [0,1])
        omega = pm.Uniform("omega", lower=0, upper=1)  # Inclusion probability
        p = pm.Uniform("p", lower=0, upper=1)  # Capture prob (not captured before)
        c = pm.Uniform("c", lower=0, upper=1)  # Capture prob (captured before)
        
        # Transformed parameters: effective capture probabilities
        # p_eff[i,j] = p if not captured on j-1, c if captured on j-1
        
        # Convert y to tensor for computation
        y_tensor = pt.as_tensor_variable(y)
        
        # Compute p_eff for all individuals and occasions
        p_eff_list = []
        
        for i in range(M):
            p_eff_i = []
            # First occasion: always p
            p_eff_i.append(p)
            
            # Subsequent occasions: depend on previous capture
            for j in range(1, T):
                # p_eff[i,j] = (1 - y[i,j-1]) * p + y[i,j-1] * c
                p_eff_ij = (1 - y_tensor[i, j-1]) * p + y_tensor[i, j-1] * c
                p_eff_i.append(p_eff_ij)
            
            p_eff_list.append(pt.stack(p_eff_i))
        
        p_eff = pt.stack(p_eff_list)  # [M, T] tensor
        
        # Likelihood
        # For each individual, marginalize over latent presence (z[i])
        logp_contributions = []
        
        for i in range(M):
            if s[i] > 0:
                # Individual was captured at least once: z[i] = 1 with certainty
                # log P(z[i]=1) + log P(y[i] | z[i]=1, p_eff[i])
                logp_z1 = pt.log(omega)
                logp_y_given_z1 = pm.Bernoulli.logp(y_tensor[i], p_eff[i]).sum()
                logp_contributions.append(logp_z1 + logp_y_given_z1)
            else:
                # Individual never captured: marginalize over z[i]
                # log[P(z[i]=1) * P(y[i]=0 | z[i]=1) + P(z[i]=0)]
                logp_z1 = pt.log(omega)
                logp_y_given_z1 = pm.Bernoulli.logp(y_tensor[i], p_eff[i]).sum()
                logp_case1 = logp_z1 + logp_y_given_z1  # z[i]=1 case
                logp_case2 = pt.log(1 - omega)  # z[i]=0 case
                logp_contributions.append(pm.math.logsumexp(pt.stack([logp_case1, logp_case2])))
        
        # Add all contributions to the model
        pm.Potential("likelihood", pt.sum(pt.stack(logp_contributions)))
        
        # Generated quantities
        omega_nd = pm.Deterministic("omega_nd", 
                                  (omega * (1 - p)**T) / (omega * (1 - p)**T + (1 - omega)))
        trap_response = pm.Deterministic("trap_response", c - p)

    return model