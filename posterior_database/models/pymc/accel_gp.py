def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    def spd_cov_exp_quad(slambda_array, sdgp, lscale_val):
        """Spectral density function of a Gaussian process"""
        # Convert to numpy array if needed
        slambda_np = np.array(slambda_array)
        # slambda_array is [NB, D] where NB is number of basis functions, D is dimension
        NB = slambda_np.shape[0]
        D = slambda_np.shape[1]
        
        # For 1D GP case (Dls == 1)
        constant = sdgp**2 * (pt.sqrt(2 * np.pi) * lscale_val) ** D
        neg_half_lscale2 = -0.5 * lscale_val**2
        dot_self_vals = pt.sum(slambda_np**2, axis=1)  # dot_self for each row
        out = constant * pt.exp(neg_half_lscale2 * dot_self_vals)
        
        return out

    def gpa(X, sdgp, lscale_val, zgp, slambda_array):
        """Compute an approximate latent Gaussian process"""
        diag_spd = pt.sqrt(spd_cov_exp_quad(slambda_array, sdgp, lscale_val))
        return pt.dot(X, diag_spd * zgp)

    with pm.Model() as model:
        # Extract data
        N = data['N']
        Y = data['Y']
        Xgp_1 = data['Xgp_1']
        slambda_1 = np.array(data['slambda_1'])  # [NBgp_1, Dgp_1]
        Xgp_sigma_1 = data['Xgp_sigma_1']
        slambda_sigma_1 = np.array(data['slambda_sigma_1'])  # [NBgp_sigma_1, Dgp_sigma_1]
        NBgp_1 = data['NBgp_1']
        NBgp_sigma_1 = data['NBgp_sigma_1']
        prior_only = data['prior_only']

        # Parameters
        Intercept = pm.StudentT("Intercept", nu=3, mu=-13, sigma=36)
        
        # GP parameters for mean
        sdgp_1 = pm.HalfStudentT("sdgp_1", nu=3, sigma=36)
        lscale_1 = pm.InverseGamma("lscale_1", alpha=1.124909, beta=0.0177)
        zgp_1 = pm.Normal("zgp_1", mu=0, sigma=1, shape=NBgp_1)
        
        # GP parameters for sigma  
        Intercept_sigma = pm.StudentT("Intercept_sigma", nu=3, mu=0, sigma=10)
        sdgp_sigma_1 = pm.HalfStudentT("sdgp_sigma_1", nu=3, sigma=36)
        lscale_sigma_1 = pm.InverseGamma("lscale_sigma_1", alpha=1.124909, beta=0.0177)
        zgp_sigma_1 = pm.Normal("zgp_sigma_1", mu=0, sigma=1, shape=NBgp_sigma_1)

        # Transformed parameters - create vector versions (Stan indexing starts at 1)
        vsdgp_1 = pm.Deterministic("vsdgp_1", pt.stack([sdgp_1]))
        vlscale_1 = pm.Deterministic("vlscale_1", pt.stack([lscale_1]))
        vsdgp_sigma_1 = pm.Deterministic("vsdgp_sigma_1", pt.stack([sdgp_sigma_1]))
        vlscale_sigma_1 = pm.Deterministic("vlscale_sigma_1", pt.stack([lscale_sigma_1]))

        # Linear predictors
        mu = Intercept + gpa(Xgp_1, vsdgp_1[0], vlscale_1[0], zgp_1, slambda_1)
        sigma_linear = Intercept_sigma + gpa(Xgp_sigma_1, vsdgp_sigma_1[0], vlscale_sigma_1[0], 
                                           zgp_sigma_1, slambda_sigma_1)
        
        # Apply inverse link function for sigma (exp)
        sigma = pm.Deterministic("sigma", pt.exp(sigma_linear))

        # Generated quantities
        b_Intercept = pm.Deterministic("b_Intercept", Intercept)
        b_sigma_Intercept = pm.Deterministic("b_sigma_Intercept", Intercept_sigma)

        # Likelihood
        if not prior_only:
            Y_obs = pm.Normal("Y", mu=mu, sigma=sigma, observed=Y)

    return model