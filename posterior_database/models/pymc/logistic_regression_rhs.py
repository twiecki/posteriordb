def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Extract data
    n = data['n']
    d = data['d']
    y = data['y']
    x = data['x']
    scale_icept = data['scale_icept']
    scale_global = data['scale_global']
    nu_global = data['nu_global']
    nu_local = data['nu_local']
    slab_scale = data['slab_scale']
    slab_df = data['slab_df']

    with pm.Model() as model:
        # Parameters
        beta0 = pm.Normal("beta0", mu=0, sigma=scale_icept)
        z = pm.Normal("z", mu=0, sigma=1, shape=d)
        
        # For the half-t priors, Stan uses student_t on positive constrained parameters
        # This maps to HalfStudentT in PyMC
        tau = pm.HalfStudentT("tau", nu=nu_global, sigma=scale_global * 2)
        lambda_ = pm.HalfStudentT("lambda", nu=nu_local, sigma=1, shape=d)
        caux = pm.InverseGamma("caux", alpha=0.5 * slab_df, beta=0.5 * slab_df)
        
        # Transformed parameters - regularized horseshoe
        c = slab_scale * pt.sqrt(caux)  # slab scale
        lambda_tilde = pt.sqrt(c**2 * lambda_**2 / (c**2 + tau**2 * lambda_**2))
        beta = pm.Deterministic("beta", z * lambda_tilde * tau)
        
        # Likelihood - bernoulli_logit_glm
        # This is equivalent to: logit(p) = beta0 + x @ beta
        logit_p = beta0 + x @ beta
        y_obs = pm.Bernoulli("y", logit_p=logit_p, observed=y)
        
        # Generated quantities (as Deterministic)
        f = pm.Deterministic("f", beta0 + x @ beta)

    return model