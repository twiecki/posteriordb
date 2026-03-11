def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    K = data['K']
    N = data['N']
    y = data['y']

    with pm.Model() as model:
        # Parameters
        # simplex[K] theta - mixture weights
        theta = pm.Dirichlet("theta", a=np.ones(K))
        
        # array[K] real mu - component means
        mu = pm.Normal("mu", mu=0, sigma=10, shape=K)
        
        # array[K] real<lower=0, upper=10> sigma - component standard deviations
        sigma = pm.Uniform("sigma", lower=0, upper=10, shape=K)
        
        # Model: Gaussian mixture
        # In Stan, this is implemented with log_sum_exp over components
        # In PyMC, we can use pm.NormalMixture directly
        y_obs = pm.NormalMixture("y", w=theta, mu=mu, sigma=sigma, observed=y)

    return model