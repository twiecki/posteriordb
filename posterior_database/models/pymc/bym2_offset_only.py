def make_model(data: dict):
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    N = data['N']
    N_edges = data['N_edges']
    node1 = np.array(data['node1']) - 1  # Convert to 0-based indexing
    node2 = np.array(data['node2']) - 1  # Convert to 0-based indexing
    y = data['y']
    E = data['E']
    scaling_factor = data['scaling_factor']
    
    # Transformed data
    log_E = np.log(E)
    
    with pm.Model() as model:
        # Parameters
        beta0 = pm.Normal("beta0", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        rho = pm.Beta("rho", alpha=0.5, beta=0.5)
        
        theta = pm.Normal("theta", mu=0, sigma=1, shape=N)
        phi = pm.Normal("phi", mu=0, sigma=1, shape=N)
        
        # Transformed parameters
        convolved_re = pt.sqrt(1 - rho) * theta + pt.sqrt(rho / scaling_factor) * phi
        
        # Spatial structure: CAR prior for phi
        # target += -0.5 * dot_self(phi[node1] - phi[node2]);
        phi_diff = phi[node1] - phi[node2]
        pm.Potential("spatial", -0.5 * pt.sum(phi_diff**2))
        
        # Sum-to-zero constraint on phi
        # sum(phi) ~ normal(0, 0.001 * N);
        # This is equivalent to a potential with the normal logpdf
        phi_sum = pt.sum(phi)
        pm.Potential("phi_sum_constraint", 
                     -0.5 * (phi_sum**2) / ((0.001 * N)**2))
        
        # Likelihood
        eta = log_E + beta0 + convolved_re * sigma
        y_obs = pm.Poisson("y", mu=pt.exp(eta), observed=y)
        
        # Generated quantities (as deterministics)
        log_precision = pm.Deterministic("log_precision", -2.0 * pt.log(sigma))
        logit_rho = pm.Deterministic("logit_rho", pm.math.logit(rho))
        eta_det = pm.Deterministic("eta", eta)
        mu = pm.Deterministic("mu", pt.exp(eta))

    return model