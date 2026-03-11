def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    with pm.Model() as model:
        # Extract data
        N = data['N']
        x = data['x']
        y = data['y']
        
        # Parameters 
        # rho ~ gamma(25, 4) - straightforward
        rho = pm.Gamma("rho", alpha=25, beta=4)
        
        # alpha and sigma are real<lower=0> with normal priors
        # This means truncated normal on [0, inf)
        alpha = pm.TruncatedNormal("alpha", mu=0, sigma=2, lower=0)
        sigma = pm.TruncatedNormal("sigma", mu=0, sigma=1, lower=0)
        
        # Build GP covariance matrix manually
        # Stan's gp_exp_quad_cov(x, alpha, rho) computes:
        # alpha^2 * exp(-0.5 * (x_i - x_j)^2 / rho^2)
        
        # Create distance matrix
        x_array = pt.as_tensor_variable(x)
        X1 = x_array[:, None]  # Shape (N, 1)
        X2 = x_array[None, :]  # Shape (1, N)
        dist_sq = (X1 - X2)**2  # Shape (N, N)
        
        # Exponentiated quadratic covariance
        K = alpha**2 * pt.exp(-0.5 * dist_sq / rho**2)
        
        # Add diagonal noise term - Stan adds diag_matrix(rep_vector(sigma, N))
        # This means adding sigma (not sigma^2) on the diagonal
        K_with_noise = K + pt.eye(N) * sigma
        
        # Stan uses multi_normal_cholesky, so compute Cholesky and use that
        L_cov = pt.linalg.cholesky(K_with_noise)
        
        # Multivariate normal likelihood with Cholesky parameterization
        y_obs = pm.MvNormal("y", mu=np.zeros(N), chol=L_cov, observed=y)
        
    return model