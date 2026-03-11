def make_model(data: dict) -> pm.Model:
    """PyMC model transpiled from Stan."""
    import pymc as pm
    import pytensor.tensor as pt
    import numpy as np
    from scipy.integrate import solve_ivp
    import pytensor
    
    # Extract data
    t0 = data['t0']
    D = data['D']
    V = data['V'] 
    times = data['times']
    N_t = data['N_t']
    C_hat = data['C_hat']
    
    # Transformed data
    C0 = np.array([0.0])
    
    class ODEOp(pytensor.tensor.Op):
        """Custom PyTensor Op for ODE integration"""
        
        def __init__(self, t0, times, C0, D, V):
            self.t0 = t0
            self.times = times
            self.C0 = C0
            self.D = D
            self.V = V
            
        def make_node(self, k_a, K_m, V_m):
            k_a = pt.as_tensor_variable(k_a)
            K_m = pt.as_tensor_variable(K_m)
            V_m = pt.as_tensor_variable(V_m)
            
            outputs = [pt.tensor(dtype='float64', shape=(len(self.times),))]
            return pytensor.graph.Apply(self, [k_a, K_m, V_m], outputs)
            
        def perform(self, node, inputs, outputs):
            k_a_val, K_m_val, V_m_val = inputs
            
            def ode_func(t, y):
                dose = 0.0
                elim = (V_m_val / self.V) * y[0] / (K_m_val + y[0])
                
                if t > 0:
                    dose = np.exp(-k_a_val * t) * self.D * k_a_val / self.V
                
                dydt = dose - elim
                return [dydt]
            
            try:
                sol = solve_ivp(ode_func, [self.t0, self.times[-1]], self.C0, 
                               t_eval=self.times, method='BDF', rtol=1e-8, atol=1e-10)
                
                if sol.success:
                    result = sol.y[0, :]
                else:
                    # Fallback if ODE solver fails
                    result = np.full(len(self.times), 1e-10)
                    
            except Exception:
                result = np.full(len(self.times), 1e-10)
                
            outputs[0][0] = result
    
    with pm.Model() as model:
        # Parameters with half-Cauchy priors (positive constraints)
        k_a = pm.HalfCauchy("k_a", beta=1.0)
        K_m = pm.HalfCauchy("K_m", beta=1.0)
        V_m = pm.HalfCauchy("V_m", beta=1.0) 
        sigma = pm.HalfCauchy("sigma", beta=1.0)
        
        # Solve ODE to get concentrations using custom Op
        ode_op = ODEOp(t0, times, C0, D, V)
        C = ode_op(k_a, K_m, V_m)
        
        # Likelihood: lognormal for observed concentrations
        # C_hat[n] ~ lognormal(log(C[n]), sigma)
        C_obs = pm.LogNormal("C_obs", mu=pt.log(C), sigma=sigma, observed=C_hat)
        
    return model