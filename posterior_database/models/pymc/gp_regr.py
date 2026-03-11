def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    N = data['N']
    x = data['x']
    y = data['y']
    
    with pm.Model() as model:
        # Parameters
        rho = pm.Gamma("rho", alpha=25, beta=4)  # Stan: gamma(25, 4)
        alpha = pm.HalfNormal("alpha", sigma=2)  # Stan: real<lower=0> alpha ~ normal(0, 2)
        sigma = pm.HalfNormal("sigma", sigma=1)  # Stan: real<lower=0> sigma ~ normal(0, 1)
        
        # Gaussian Process covariance matrix
        # gp_exp_quad_cov(x, alpha, rho) in Stan
        x_arr = np.array(x, dtype=float)
        
        # Create distance matrix
        x_reshaped = x_arr.reshape(-1, 1)  # Shape: (N, 1)
        sq_dist = (x_reshaped - x_reshaped.T)**2  # Shape: (N, N)
        
        # GP covariance: alpha^2 * exp(-0.5 * sq_dist / rho^2)
        cov_gp = alpha**2 * pt.exp(-0.5 * sq_dist / rho**2)
        
        # Add noise to diagonal: + diag_matrix(rep_vector(sigma, N))
        # Stan's diag_matrix(rep_vector(sigma, N)) creates a diagonal matrix with sigma on diagonal
        # So this adds sigma (not sigma^2) to each diagonal element
        cov = cov_gp + pt.diag(pt.full(N, sigma))
        
        # Cholesky decomposition
        L_cov = pt.linalg.cholesky(cov)
        
        # Likelihood: y ~ multi_normal_cholesky(rep_vector(0, N), L_cov)
        y_obs = pm.MvNormal("y", mu=pt.zeros(N), chol=L_cov, observed=y)
        
        # Account for half-distribution log(2) correction
        # We have 2 half-normal distributions: alpha and sigma
        pm.Potential("half_dist_correction", -2 * pt.log(2.0))
    
    return model