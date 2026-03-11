def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    J = data['J']  # sites within region
    K = data['K']  # visits to sites  
    n = data['n']  # observed species
    X = np.array(data['X'])  # Convert to numpy array to ensure proper indexing
    S = data['S']  # superpopulation size

    with pm.Model() as model:
        
        # Priors
        alpha = pm.Cauchy("alpha", alpha=0, beta=2.5)
        beta = pm.Cauchy("beta", alpha=0, beta=2.5)
        
        # Availability probability
        Omega = pm.Beta("Omega", alpha=2, beta=2)
        
        # Correlation parameter: (rho_uv + 1) / 2 ~ beta(2, 2)
        rho_uv_scaled = pm.Beta("rho_uv_scaled", alpha=2, beta=2)
        rho_uv = pm.Deterministic("rho_uv", 2 * rho_uv_scaled - 1)
        
        # Standard deviations for bivariate normal
        sigma_uv = pm.HalfCauchy("sigma_uv", beta=2.5, shape=2)
        
        # Species-level random effects as separate vectors (matching Stan)
        uv1 = pm.Normal("uv1", mu=0, sigma=1, shape=S)
        uv2 = pm.Normal("uv2", mu=0, sigma=1, shape=S)
        
        # Build covariance matrix components
        Sigma_11 = sigma_uv[0]**2
        Sigma_22 = sigma_uv[1]**2  
        Sigma_12 = sigma_uv[0] * sigma_uv[1] * rho_uv
        
        # Add the multivariate normal constraint
        # The Stan model does: target += multi_normal_lpdf(uv | rep_vector(0, 2), cov_matrix_2d(...))
        # where uv[i] = [uv1[i], uv2[i]] for each i
        
        det_Sigma = Sigma_11 * Sigma_22 - Sigma_12**2
        inv_Sigma_11 = Sigma_22 / det_Sigma
        inv_Sigma_12 = -Sigma_12 / det_Sigma  
        inv_Sigma_22 = Sigma_11 / det_Sigma
        
        mvn_logp = 0.0
        for i in range(S):
            u1_i = uv1[i]
            u2_i = uv2[i] 
            
            # Quadratic form: u^T Sigma^{-1} u
            quad_form = (u1_i * inv_Sigma_11 * u1_i + 
                        2 * u1_i * inv_Sigma_12 * u2_i +
                        u2_i * inv_Sigma_22 * u2_i)
            
            # Log determinant term
            log_det_term = 0.5 * pt.log(det_Sigma)
            
            mvn_logp += -0.5 * quad_form - log_det_term - np.log(2 * np.pi)
        
        # Subtract the independent normal contributions we already included
        independent_logp = pt.sum(pm.Normal.logp(uv1, mu=0, sigma=1)) + pt.sum(pm.Normal.logp(uv2, mu=0, sigma=1))
        pm.Potential("mvn_correction", mvn_logp - independent_logp)
        
        # Transformed parameters
        logit_psi = pm.Deterministic("logit_psi", uv1 + alpha)
        logit_theta = pm.Deterministic("logit_theta", uv2 + beta)
        
        # Convert logits to probabilities for use in binomial
        psi = pm.math.invlogit(logit_psi)
        theta = pm.math.invlogit(logit_theta)
        
        # Likelihood computation
        total_logp = 0.0
        
        # For observed species (indices 0 to n-1)
        for i in range(n):
            # Species i is available
            total_logp += pm.Bernoulli.logp(1, p=Omega)
            
            # For each site
            for j in range(J):
                X_ij = X[i, j]
                if X_ij > 0:
                    # lp_observed: log(psi) + binomial_logit_lpmf(X | K, logit_theta)
                    total_logp += (pt.log(psi[i]) + 
                                 pm.Binomial.logp(X_ij, n=K, p=theta[i]))
                else:
                    # lp_unobserved: log_sum_exp(lp_observed(0, ...), log1m_inv_logit(logit_psi))
                    lp_obs_zero = pt.log(psi[i]) + pm.Binomial.logp(0, n=K, p=theta[i])
                    lp_not_present = pt.log(1 - psi[i])  # log1m_inv_logit(logit_psi)
                    total_logp += pm.math.logsumexp(pt.stack([lp_obs_zero, lp_not_present]))
        
        # For never observed species (indices n to S-1) 
        for i in range(n, S):
            # lp_never_observed
            lp_unavailable = pm.Bernoulli.logp(0, p=Omega)
            
            # Compute lp_unobserved for this species
            lp_obs_zero_i = pt.log(psi[i]) + pm.Binomial.logp(0, n=K, p=theta[i])
            lp_not_present_i = pt.log(1 - psi[i])
            lp_unobs_i = pm.math.logsumexp(pt.stack([lp_obs_zero_i, lp_not_present_i]))
            
            lp_available = pm.Bernoulli.logp(1, p=Omega) + J * lp_unobs_i
            total_logp += pm.math.logsumexp(pt.stack([lp_unavailable, lp_available]))
        
        # Add likelihood
        pm.Potential("likelihood", total_logp)
        
        # Derived quantities
        E_N = pm.Deterministic("E_N", S * Omega)

    return model