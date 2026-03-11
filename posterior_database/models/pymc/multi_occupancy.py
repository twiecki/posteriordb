def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        J = data['J']  # sites within region
        K = data['K']  # visits to sites  
        n = data['n']  # observed species
        X = np.array(data['X'])  # visits when species i was detected at site j
        S = data['S']  # superpopulation size
        
        # Parameters
        alpha = pm.Cauchy("alpha", alpha=0, beta=2.5)  # site-level occupancy
        beta = pm.Cauchy("beta", alpha=0, beta=2.5)   # site-level detection
        Omega = pm.Beta("Omega", alpha=2, beta=2)     # availability of species
        
        # Correlation parameter with special prior: (rho_uv + 1)/2 ~ beta(2, 2)
        rho_uv_scaled = pm.Beta("rho_uv_scaled", alpha=2, beta=2)
        rho_uv = pm.Deterministic("rho_uv", 2 * rho_uv_scaled - 1)
        
        # Scale parameters
        sigma_uv = pm.HalfCauchy("sigma_uv", beta=2.5, shape=2)
        
        # Species-level random effects 
        # Stan declares these as vector[S] uv1, vector[S] uv2, then applies MVN structure
        uv1 = pm.Normal("uv1", mu=0, sigma=1, shape=S)
        uv2 = pm.Normal("uv2", mu=0, sigma=1, shape=S)
        
        # Apply multivariate normal structure via potential
        # Stan: target += multi_normal_lpdf(uv | rep_vector(0, 2), cov_matrix_2d(sigma_uv, rho_uv))
        mvn_potential = 0.0
        
        for i in range(S):
            # For each species i, we have [uv1[i], uv2[i]] ~ MVN(0, Sigma)
            # where Sigma = [[sigma1^2, rho*sigma1*sigma2], [rho*sigma1*sigma2, sigma2^2]]
            
            u1 = uv1[i]
            u2 = uv2[i] 
            
            # Covariance matrix elements
            var1 = sigma_uv[0]**2
            var2 = sigma_uv[1]**2
            cov12 = rho_uv * sigma_uv[0] * sigma_uv[1]
            
            # Determinant and inverse
            det_sigma = var1 * var2 - cov12**2
            inv_11 = var2 / det_sigma
            inv_22 = var1 / det_sigma
            inv_12 = -cov12 / det_sigma
            
            # Quadratic form
            quad_form = inv_11 * u1**2 + inv_22 * u2**2 + 2 * inv_12 * u1 * u2
            
            # MVN log density
            mvn_potential += -0.5 * quad_form - 0.5 * pt.log(det_sigma) - np.log(2 * np.pi)
            
            # Subtract the standard normal densities that are already counted
            mvn_potential -= -0.5 * u1**2 - 0.5 * np.log(2 * np.pi)
            mvn_potential -= -0.5 * u2**2 - 0.5 * np.log(2 * np.pi)
        
        pm.Potential("mvn_structure", mvn_potential)
        
        # Transformed parameters - match Stan exactly
        logit_psi = uv1 + alpha  # Stan: logit_psi[i] = uv[i, 1] + alpha
        logit_theta = uv2 + beta  # Stan: logit_theta[i] = uv[i, 2] + beta
        
        # Likelihood components
        total_logp = 0.0
        
        # Helper functions
        def log_inv_logit(x):
            return x - pt.log1p(pt.exp(x))
        
        def log1m_inv_logit(x):
            return -pt.log1p(pt.exp(x))
        
        def binomial_logit_lpmf(k, n_trials, logit_p):
            # binomial(k | n, p) with p = invlogit(logit_p)
            return (k * logit_p - n_trials * pt.log1p(pt.exp(logit_p)) +
                   pt.gammaln(n_trials + 1) - pt.gammaln(k + 1) - pt.gammaln(n_trials - k + 1))
        
        def lp_observed(x_val, k_val, logit_psi_val, logit_theta_val):
            return log_inv_logit(logit_psi_val) + binomial_logit_lpmf(x_val, k_val, logit_theta_val)
        
        def lp_unobserved(k_val, logit_psi_val, logit_theta_val):
            lp_obs_0 = lp_observed(0, k_val, logit_psi_val, logit_theta_val)
            lp_not_psi = log1m_inv_logit(logit_psi_val)
            return pm.math.logaddexp(lp_obs_0, lp_not_psi)
        
        # Observed species likelihood
        for i in range(n):
            # 1 ~ bernoulli(Omega) - observed species are available
            total_logp += pt.log(Omega)
            
            for j in range(J):
                x_ij = X[i, j]
                if x_ij > 0:
                    total_logp += lp_observed(x_ij, K, logit_psi[i], logit_theta[i])
                else:
                    total_logp += lp_unobserved(K, logit_psi[i], logit_theta[i])
        
        # Never observed species likelihood  
        for i in range(n, S):
            lp_unavailable = pt.log(1 - Omega)  # bernoulli_lpmf(0 | Omega)
            lp_available = pt.log(Omega) + J * lp_unobserved(K, logit_psi[i], logit_theta[i])
            total_logp += pm.math.logaddexp(lp_unavailable, lp_available)
        
        pm.Potential("likelihood", total_logp)
        
        # Add correction for HalfCauchy distributions
        pm.Potential("half_dist_correction", -2 * pt.log(2.0))
    
    return model