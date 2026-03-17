def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np

    with pm.Model() as model:
        # Extract data
        weight = np.array(data['weight'])
        diam1 = np.array(data['diam1'])
        diam2 = np.array(data['diam2'])
        canopy_height = np.array(data['canopy_height'])
        group = np.array(data['group'])

        # Transformed data
        log_weight = np.log(weight)
        log_canopy_volume = np.log(diam1 * diam2 * canopy_height)
        log_canopy_area = np.log(diam1 * diam2)

        # Parameters with weakly informative priors (equivalent to flat for this data scale)
        beta = pm.Normal("beta", mu=0, sigma=100, shape=4)
        sigma = pm.HalfNormal("sigma", sigma=100)

        # Model
        mu = beta[0] + beta[1] * log_canopy_volume + beta[2] * log_canopy_area + beta[3] * group

        pm.Normal("log_weight", mu=mu, sigma=sigma, observed=log_weight)

    return model
