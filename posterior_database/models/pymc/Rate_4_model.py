def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Parameters
        # Beta(1,1) is equivalent to Uniform(0,1)
        theta = pm.Uniform("theta", lower=0, upper=1)
        thetaprior = pm.Uniform("thetaprior", lower=0, upper=1)
        
        # Observed data - k follows binomial(n, theta)
        k_obs = pm.Binomial("k", n=data['n'], p=theta, observed=data['k'])
        
        # Stan uses proportional form, so we need to subtract the binomial coefficient
        # log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
        # Using gammaln since log(n!) = gammaln(n+1)
        n = data['n']
        k = data['k']
        log_binom_coeff = pt.gammaln(n + 1) - pt.gammaln(k + 1) - pt.gammaln(n - k + 1)

    return model