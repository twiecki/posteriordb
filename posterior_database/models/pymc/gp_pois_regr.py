def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    N = data['N']
    x = np.array(data['x'], dtype=float)
    k = np.array(data['k'])

    with pm.Model() as model:
        
        # Parameters with proper constraints
        rho = pm.Gamma("rho", alpha=25, beta=4)
        
        # For alpha: real<lower=0> with normal(0, 2) prior
        # This is equivalent to a HalfNormal but with log(2) offset
        alpha = pm.HalfNormal("alpha", sigma=2)
        
        f_tilde = pm.Normal("f_tilde", mu=0, sigma=1, shape=N)
        
        # Transformed parameters
        # Convert x to tensor
        x_tensor = pt.as_tensor_variable(x)
        
        # Compute GP covariance matrix using Stan's gp_exp_quad_cov parameterization
        x_diff = x_tensor[:, None] - x_tensor[None, :]  # Broadcasting to get all pairwise differences
        
        # Stan's gp_exp_quad_cov: alpha^2 * exp(-0.5 * sqdist / rho^2)
        sqdist = x_diff ** 2
        cov = alpha**2 * pt.exp(-0.5 * sqdist / rho**2)
        
        # Add jitter for numerical stability
        cov = cov + pt.eye(N) * 1e-10
        
        # Cholesky decomposition
        L_cov = pt.linalg.cholesky(cov)
        
        # Transform f_tilde to get f
        f = pm.Deterministic("f", L_cov @ f_tilde)
        
        # Likelihood - Stan uses poisson_log which means log-parameterization
        k_obs = pm.Poisson("k", mu=pt.exp(f), observed=k)
    
    return model