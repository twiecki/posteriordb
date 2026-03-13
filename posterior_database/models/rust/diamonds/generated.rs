use std::collections::HashMap;
use nuts_rs::{CpuLogpFunc, CpuMathError, LogpError, Storable};
use nuts_storable::HasDims;
use thiserror::Error;
use crate::data::*;

#[derive(Debug, Error)]
pub enum SampleError {
    #[error("Recoverable: {0}")]
    Recoverable(String),
}

impl LogpError for SampleError {
    fn is_recoverable(&self) -> bool { true }
}

pub const N_PARAMS: usize = 26;

const LN_2PI: f64 = 1.8378770664093453;

#[derive(Storable, Clone)]
pub struct Draw {
    #[storable(dims("param"))]
    pub parameters: Vec<f64>,
}

#[derive(Clone, Default)]
pub struct GeneratedLogp;

impl HasDims for GeneratedLogp {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([("param".to_string(), N_PARAMS as u64)])
    }
}

impl CpuLogpFunc for GeneratedLogp {
    type LogpError = SampleError;
    type FlowParameters = ();
    type ExpandedVector = Draw;

    fn dim(&self) -> usize { N_PARAMS }

    fn logp(&mut self, position: &[f64], gradient: &mut [f64]) -> Result<f64, SampleError> {
        // Zero the gradient
        for i in 0..N_PARAMS {
            gradient[i] = 0.0;
        }

        let mut logp = 0.0;

        // Extract parameters
        let b = &position[0..24];  // b ~ Normal(0, 1), shape [24]
        let intercept = position[24];  // Intercept ~ StudentT(nu=3, mu=8, sigma=10)
        let log_sigma = position[25];  // sigma_log__ (log-transformed)
        let sigma = log_sigma.exp();

        // Prior: b ~ Normal(0, 1) for each component
        // logp = -0.5*log(2*pi) - log(1) - 0.5*((b-0)/1)^2
        // logp = -0.5*log(2*pi) - 0.5*b^2
        for i in 0..24 {
            let b_i = b[i];
            logp += -0.5 * LN_2PI - 0.5 * b_i * b_i;
            gradient[i] = -b_i;  // d/db_i = -b_i
        }

        // Prior: Intercept ~ StudentT(nu=3, mu=8, sigma=10)
        // Using the formula from PyMC graph: -3.303473950203429 - (2.0 * log1p((0.003333333258827527 * sqr((-8.0 + i0)))))
        let intercept_centered = intercept - 8.0;
        let nu = 3.0;
        let scale_intercept = 10.0;
        let scale_intercept_sq = scale_intercept * scale_intercept;
        
        // StudentT logp: constant - (nu+1)/2 * log(1 + t^2/(nu*scale^2))
        // where t = (x - mu) and the constant includes normalization
        let t_sq_scaled = intercept_centered * intercept_centered / (nu * scale_intercept_sq);
        let log_term = (1.0 + t_sq_scaled).ln();
        let intercept_logp = -3.303473950203429 - 2.0 * log_term;
        logp += intercept_logp;
        
        // Gradient: d/d(intercept) = -(nu+1)/nu * (x-mu)/(scale^2 + (x-mu)^2/nu)
        let denom = nu * scale_intercept_sq + intercept_centered * intercept_centered;
        gradient[24] = -4.0 * intercept_centered / denom;

        // Prior: sigma ~ HalfStudentT(nu=3, sigma=10) with LogTransform
        // From PyMC graph: switch(lt(exp(i0), 0), -inf, (-2.610326741661797 - (2.0 * log1p((0.003333333333333333 * sqr(exp(i0))))))) + i0
        if sigma <= 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        
        let nu_sigma = 3.0;
        let scale_sigma = 10.0;
        let scale_sigma_sq = scale_sigma * scale_sigma;
        
        // HalfStudentT logp + Jacobian
        let sigma_sq_scaled = sigma * sigma / (nu_sigma * scale_sigma_sq);
        let log_term_sigma = (1.0 + sigma_sq_scaled).ln();
        let sigma_logp = -2.610326741661797 - 2.0 * log_term_sigma + log_sigma;  // +log_sigma is Jacobian
        logp += sigma_logp;
        
        // Gradient: d/d(log_sigma) = d/d(sigma) * sigma + 1 (Jacobian term)
        let denom_sigma = nu_sigma * scale_sigma_sq + sigma * sigma;
        let d_sigma = -4.0 * sigma / denom_sigma;
        gradient[25] = d_sigma * sigma + 1.0;  // Chain rule + Jacobian

        // Likelihood: Y ~ Normal(Intercept, sigma)
        // This is the dominant term
        let inv_sigma = 1.0 / sigma;
        let inv_sigma_sq = inv_sigma * inv_sigma;
        let log_norm_const = -0.5 * LN_2PI - log_sigma;  // use log_sigma directly instead of sigma.ln()
        
        let mut sum_residuals = 0.0;
        let mut sum_residuals_sq = 0.0;
        
        for i in 0..Y_N {
            let residual = Y_DATA[i] - intercept;
            sum_residuals += residual;
            sum_residuals_sq += residual * residual;
            logp += log_norm_const - 0.5 * residual * residual * inv_sigma_sq;
        }
        
        // Gradients for likelihood
        // d/d(intercept) = sum((y_i - intercept) / sigma^2)
        gradient[24] += sum_residuals * inv_sigma_sq;
        
        // d/d(log_sigma) = -N + sum((y_i - intercept)^2) / sigma^2
        gradient[25] += -(Y_N as f64) + sum_residuals_sq * inv_sigma_sq;

        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}