def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    N = data['N']
    income = np.array(data['income'])
    vote = np.array(data['vote'])
    x = income.reshape(-1, 1)

    with pm.Model() as model:
        # Parameters with improper priors (Stan has no explicit priors)
        alpha = pm.Flat("alpha")
        # Treat beta as scalar since it's vector[1] in Stan
        beta = pm.Flat("beta")
        
        # Model: vote ~ bernoulli_logit_glm(x, alpha, beta)
        # This is equivalent to: vote ~ bernoulli_logit(alpha + x * beta)
        logit_p = alpha + (x * beta).flatten()
        
        vote_obs = pm.Bernoulli("vote", logit_p=logit_p, observed=vote)

    return model