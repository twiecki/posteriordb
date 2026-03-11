def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    
    # Extract data
    T = data['T']
    K = data['K']
    t = data['t']
    cap = data['cap']
    y = data['y']
    S = data['S']
    t_change = np.array(data['t_change'])
    X = data['X']
    sigmas = np.array(data['sigmas'])
    tau = data['tau']
    trend_indicator = data['trend_indicator']
    s_a = np.array(data['s_a'])
    s_m = np.array(data['s_m'])
    
    # Transformed data: compute changepoint matrix A
    def get_changepoint_matrix(t, t_change, T, S):
        A = np.zeros((T, S))
        for i in range(T):
            for j in range(S):
                if t[i] >= t_change[j]:
                    A[i, j] = 1.0
        return A
    
    A = get_changepoint_matrix(t, t_change, T, S)
    
    # Helper functions
    def logistic_gamma(k, m, delta, t_change, S):
        # Compute the rate in each segment
        k_s = pt.concatenate([pt.stack([k]), k + pt.cumsum(delta)])
        
        # Piecewise offsets
        gamma = pt.zeros(S)
        m_pr = m
        for i in range(S):
            gamma_i = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1])
            gamma = pt.set_subtensor(gamma[i], gamma_i)
            m_pr = m_pr + gamma_i
        return gamma
    
    def logistic_trend(k, m, delta, t, cap, A, t_change, S):
        gamma = logistic_gamma(k, m, delta, t_change, S)
        rate = k + pt.dot(A, delta)
        offset = m + pt.dot(A, gamma)
        return cap * pm.math.invlogit(rate * (t - offset))
    
    def linear_trend(k, m, delta, t, A, t_change):
        neg_t_change = -t_change
        return (k + pt.dot(A, delta)) * t + (m + pt.dot(A, neg_t_change * delta))
    
    with pm.Model() as model:
        # Parameters
        k = pm.Normal("k", mu=0, sigma=5)
        m = pm.Normal("m", mu=0, sigma=5)
        delta = pm.Laplace("delta", mu=0, b=tau, shape=S)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)
        beta = pm.Normal("beta", mu=0, sigma=sigmas, shape=K)
        
        # Likelihood based on trend_indicator
        if trend_indicator == 0:  # Linear trend
            trend = linear_trend(k, m, delta, t, A, t_change)
        else:  # Logistic trend
            trend = logistic_trend(k, m, delta, t, cap, A, t_change, S)
        
        # Combine trend with additive and multiplicative features
        mu = trend * (1 + pt.dot(X, beta * s_m)) + pt.dot(X, beta * s_a)
        
        # Observation likelihood
        y_obs = pm.Normal("y", mu=mu, sigma=sigma_obs, observed=y)
    
    return model