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

pub const N_PARAMS: usize = 6;

const LN_2PI: f64 = 1.8378770664093453;
const LN_2: f64 = 0.6931471805599453;

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
        // Extract parameters
        let beta = &position[0..5];  // beta[0..4] 
        let log_sigma = position[5]; // log-transformed sigma
        let sigma = log_sigma.exp();
        
        // Clear gradient
        for i in 0..N_PARAMS {
            gradient[i] = 0.0;
        }
        
        let mut logp = 0.0;
        
        // ─── Prior: beta ~ Normal(0, 10) ───
        // Each component: logp += -0.5*ln(2π) - ln(10) - 0.5*(beta_i/10)^2
        let beta_log_norm = -0.5 * LN_2PI - 10.0_f64.ln();  // -3.221523658174155
        let inv_beta_var = 1.0 / (10.0 * 10.0);  // 1/100 = 0.01
        
        for i in 0..5 {
            let beta_i = beta[i];
            logp += beta_log_norm - 0.5 * beta_i * beta_i * inv_beta_var;
            gradient[i] -= beta_i * inv_beta_var;  // d/d(beta_i) = -beta_i/100
        }
        
        // ─── Prior: sigma ~ HalfNormal(10) with LogTransform + Jacobian ───
        // HalfNormal(sigma | 10): logp = ln(2) - 0.5*ln(2π) - ln(10) - 0.5*(sigma/10)^2
        // LogTransform Jacobian: +log_sigma
        let sigma_scaled = sigma / 10.0;
        let sigma_prior_log = LN_2 - 0.5 * LN_2PI - 10.0_f64.ln() - 0.5 * sigma_scaled * sigma_scaled + log_sigma;
        logp += sigma_prior_log;
        
        // Gradient w.r.t log_sigma:
        // d/d(log_sigma) = -sigma^2/100 + 1 = -sigma_scaled^2 + 1
        gradient[5] += -sigma_scaled * sigma_scaled + 1.0;
        
        // ─── Likelihood: y ~ Normal(X*beta, sigma) ───
        // We need to compute X*beta where X is 100x5 and beta is 5x1
        // X is stored as X_0_DATA with shape (500,) representing a flattened 100x5 matrix
        
        let inv_sigma = 1.0 / sigma;
        let inv_sigma_sq = inv_sigma * inv_sigma;
        let obs_log_norm = -0.5 * LN_2PI - log_sigma;  // -0.5*ln(2π) - ln(sigma)
        
        // Gradient accumulators for efficiency
        let mut grad_beta = [0.0_f64; 5];
        let mut grad_log_sigma = 0.0_f64;
        
        for i in 0..Y_N {  // Y_N = 100 observations
            // Compute mu_i = X[i,:] * beta
            let mut mu_i = 0.0;
            for j in 0..5 {
                let x_ij = X_0_DATA[i * 5 + j];  // X[i,j] from flattened array
                mu_i += x_ij * beta[j];
            }
            
            let residual = Y_DATA[i] - mu_i;
            let scaled_residual = residual * inv_sigma_sq;
            
            // Log-likelihood contribution
            logp += obs_log_norm - 0.5 * residual * residual * inv_sigma_sq;
            
            // Gradient contributions
            for j in 0..5 {
                let x_ij = X_0_DATA[i * 5 + j];
                grad_beta[j] += scaled_residual * x_ij;
            }
            grad_log_sigma += scaled_residual * residual - 1.0;
        }
        
        // Add accumulated gradients
        for j in 0..5 {
            gradient[j] += grad_beta[j];
        }
        gradient[5] += grad_log_sigma;
        
        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}