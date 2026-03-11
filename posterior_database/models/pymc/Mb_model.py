def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    y_data = data['y']  # Shape [M, T] - numpy array
    M = data['M'] 
    T = data['T']
    
    # Transformed data (compute before model)
    s = np.sum(y_data, axis=1)  # Row sums
    observed_mask = s > 0  # Boolean mask for observed individuals
    
    with pm.Model() as model:
        # Convert data to pytensor constant
        y = pt.as_tensor_variable(y_data)
        
        # Parameters with uniform priors (implicit in Stan)
        omega = pm.Uniform("omega", lower=0, upper=1)
        p = pm.Uniform("p", lower=0, upper=1)
        c = pm.Uniform("c", lower=0, upper=1)
        
        # Compute effective capture probabilities
        # Initialize with first occasion probabilities (all p)
        p_eff = pt.zeros((M, T))
        p_eff = pt.set_subtensor(p_eff[:, 0], p)
        
        # For subsequent occasions, compute iteratively
        for j in range(1, T):
            # p_eff[i,j] = (1 - y[i,j-1]) * p + y[i,j-1] * c
            p_eff_j = (1.0 - y[:, j-1]) * p + y[:, j-1] * c
            p_eff = pt.set_subtensor(p_eff[:, j], p_eff_j)
        
        # Custom likelihood
        logp_components = []
        
        # For each individual
        for i in range(M):
            if observed_mask[i]:
                # Individual was observed: z[i] = 1
                logp_omega = pt.log(omega)
                # Bernoulli log pmf for capture history
                y_i = y[i, :]  # Individual i's capture history
                p_eff_i = p_eff[i, :]  # Individual i's effective probabilities
                logp_captures = pt.sum(y_i * pt.log(p_eff_i) + (1 - y_i) * pt.log(1 - p_eff_i))
                logp_components.append(logp_omega + logp_captures)
            else:
                # Individual never observed: marginalize over z[i]
                y_i = y[i, :]
                p_eff_i = p_eff[i, :]
                
                # Case z[i] = 1: individual present
                logp_z1 = pt.log(omega) + pt.sum(y_i * pt.log(p_eff_i) + (1 - y_i) * pt.log(1 - p_eff_i))
                
                # Case z[i] = 0: individual absent
                logp_z0 = pt.log(1 - omega)
                
                # log_sum_exp(logp_z1, logp_z0)
                max_logp = pt.maximum(logp_z1, logp_z0)
                log_sum_exp = max_logp + pt.log(pt.exp(logp_z1 - max_logp) + pt.exp(logp_z0 - max_logp))
                logp_components.append(log_sum_exp)
        
        # Sum all log probability components
        total_logp = pt.sum(pt.stack(logp_components))
        pm.Potential("likelihood", total_logp)
        
        # Generated quantities (deterministic transformations)
        omega_nd = pm.Deterministic(
            "omega_nd",
            (omega * (1 - p) ** T) / (omega * (1 - p) ** T + (1 - omega))
        )
        
        trap_response = pm.Deterministic("trap_response", c - p)
    
    return model