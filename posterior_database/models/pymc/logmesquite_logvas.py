def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    # Convert data to numpy arrays
    weight = np.array(data['weight'])
    diam1 = np.array(data['diam1'])
    diam2 = np.array(data['diam2'])
    canopy_height = np.array(data['canopy_height'])
    total_height = np.array(data['total_height'])
    density = np.array(data['density'], dtype=float)
    group = np.array(data['group'], dtype=float)

    # Transformed data - compute derived quantities from input data
    log_weight = np.log(weight)
    log_canopy_volume = np.log(diam1 * diam2 * canopy_height)
    log_canopy_area = np.log(diam1 * diam2)
    log_canopy_shape = np.log(diam1 / diam2)
    log_total_height = np.log(total_height)
    log_density = np.log(density)

    with pm.Model() as model:
        # Parameters with weakly informative priors (equivalent to flat for this data scale)
        beta = pm.Normal("beta", mu=0, sigma=100, shape=7)
        sigma = pm.HalfNormal("sigma", sigma=100)

        # Linear predictor
        mu = (beta[0] +
              beta[1] * log_canopy_volume +
              beta[2] * log_canopy_area +
              beta[3] * log_canopy_shape +
              beta[4] * log_total_height +
              beta[5] * log_density +
              beta[6] * group)

        # Likelihood
        pm.Normal("log_weight", mu=mu, sigma=sigma, observed=log_weight)

    return model
