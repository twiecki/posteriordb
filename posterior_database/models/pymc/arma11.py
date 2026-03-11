def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Data
        T = data['T']
        y = data['y']
        
        # Priors
        mu = pm.Normal("mu", mu=0, sigma=10)
        phi = pm.Normal("phi", mu=0, sigma=2)
        theta = pm.Normal("theta", mu=0, sigma=2)
        sigma = pm.HalfCauchy("sigma", beta=2.5)
        
        # Compute the ARMA(1,1) log-likelihood using a custom potential
        def arma_logp(y_vals, mu, phi, theta, sigma):
            T = len(y_vals)
            
            # Initialize
            nu_1 = mu + phi * mu  # nu[1] = mu + phi * mu (assume err[0] == 0)
            err_1 = y_vals[0] - nu_1
            logp = -0.5 * (err_1 / sigma) ** 2 - pt.log(sigma) - 0.5 * pt.log(2 * np.pi)
            
            # Sequential computation
            err_prev = err_1
            for t in range(1, T):
                nu_t = mu + phi * y_vals[t-1] + theta * err_prev
                err_t = y_vals[t] - nu_t
                logp += -0.5 * (err_t / sigma) ** 2 - pt.log(sigma) - 0.5 * pt.log(2 * np.pi)
                err_prev = err_t
                
            return logp
        
        # Add the ARMA likelihood as a potential
        pm.Potential("arma_likelihood", arma_logp(y, mu, phi, theta, sigma))

    return model