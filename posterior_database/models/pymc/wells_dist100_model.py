def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    N = data['N']
    switched = data['switched']
    dist = np.array(data['dist'])
    dist100 = dist / 100.0

    with pm.Model() as model:
        # Parameters
        alpha = pm.Flat("alpha")
        # Create beta as scalar even though it's vector[1] in Stan
        beta = pm.Flat("beta")
        
        # Model: bernoulli_logit_glm
        # In Stan: bernoulli_logit_glm(x, alpha, beta)
        # This is equivalent to: logit(p) = alpha + x * beta
        logit_p = alpha + dist100 * beta
        
        # Likelihood
        switched_obs = pm.Bernoulli("switched", logit_p=logit_p, observed=switched)

    return model