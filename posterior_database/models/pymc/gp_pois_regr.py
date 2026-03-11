def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    N = data['N']
    x = np.array(data['x'], dtype=float)
    k = np.array(data['k'])
    
    with pm.Model() as model:
        # Parameters
        rho = pm.Gamma("rho", alpha=25, beta=4)
        alpha = pm.HalfNormal("alpha", sigma=2)  # alpha is constrained to be positive
        f_tilde = pm.Normal("f_tilde", mu=0, sigma=1, shape=N)
        
        # Transformed parameters - GP computation
        # Compute squared distances between all pairs of x values
        x_diff = x[:, None] - x[None, :]  # Shape: (N, N)
        sq_distances = x_diff**2
        
        # Stan's gp_exp_quad_cov uses: alpha^2 * exp(-0.5 * d^2 / rho^2)
        # where d is the distance between points
        cov = alpha**2 * pt.exp(-0.5 * sq_distances / rho**2) + pt.eye(N) * 1e-10
        
        # Cholesky decomposition and non-centered parameterization
        L_cov = pt.linalg.cholesky(cov)
        f = pm.Deterministic("f", pt.dot(L_cov, f_tilde))
        
        # Likelihood - Stan uses poisson_log which is Poisson with log link
        k_obs = pm.Poisson("k", mu=pt.exp(f), observed=k)
        
    return model