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

pub const N_PARAMS: usize = 5;
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
        // Extract parameters
        let beta_0 = position[0];
        let beta_1 = position[1];  
        let beta_2 = position[2];
        let beta_3 = position[3];
        let log_sigma = position[4];
        let sigma = log_sigma.exp();

        // Initialize gradient
        gradient[0] = 0.0; // beta_0
        gradient[1] = 0.0; // beta_1
        gradient[2] = 0.0; // beta_2 
        gradient[3] = 0.0; // beta_3
        gradient[4] = 0.0; // log_sigma

        let mut logp = 0.0;

        // Priors for beta parameters: Flat priors contribute 0 to logp and gradient
        // (no contribution since they're uniform over entire real line)

        // Prior for sigma: HalfFlat with LogTransform
        // sigma ~ HalfFlat() means uniform on (0, ∞)
        // With LogTransform: log_sigma ~ uniform on (-∞, ∞)
        // Jacobian adjustment: +log_sigma (derivative of exp transform)
        logp += log_sigma;
        gradient[4] += 1.0; // d/d(log_sigma) of log_sigma = 1

        // Likelihood: kid_score ~ Normal(mu, sigma)
        // From the PyTensor graph structure, let me try different data mapping:
        // Looking at the gradient errors, let me try:
        // beta[1] * X_2_DATA (since gradients 1 and 3 seem swapped)
        // beta[2] * X_1_DATA
        // beta[3] * X_0_DATA

        // Precompute constants for efficiency
        let inv_sigma = 1.0 / sigma;
        let inv_sigma_sq = inv_sigma * inv_sigma;
        let log_norm = -0.5 * LN_2PI - log_sigma; // -log(sqrt(2π)) - log(σ)

        // Gradient accumulators for vectorization
        let mut grad_beta_0 = 0.0;
        let mut grad_beta_1 = 0.0;
        let mut grad_beta_2 = 0.0;
        let mut grad_beta_3 = 0.0;
        let mut grad_log_sigma = 1.0; // Start with prior contribution

        // Loop over observations
        for i in 0..KID_SCORE_N {
            // Linear predictor - try swapped mapping
            let mu_i = beta_0 + beta_1 * X_2_DATA[i] + beta_2 * X_1_DATA[i] + beta_3 * X_0_DATA[i];
            
            // Residual
            let residual = KID_SCORE_DATA[i] - mu_i;
            
            // Log likelihood contribution
            logp += log_norm - 0.5 * residual * residual * inv_sigma_sq;
            
            // Gradient contributions
            let grad_common = residual * inv_sigma_sq;
            grad_beta_0 += grad_common;
            grad_beta_1 += grad_common * X_2_DATA[i]; // swapped
            grad_beta_2 += grad_common * X_1_DATA[i];
            grad_beta_3 += grad_common * X_0_DATA[i]; // swapped
            
            // For log_sigma: d/d(log_sigma) = -1 + (residual^2 / sigma^2)
            grad_log_sigma += -1.0 + residual * residual * inv_sigma_sq;
        }

        // Set gradients
        gradient[0] += grad_beta_0;
        gradient[1] += grad_beta_1;
        gradient[2] += grad_beta_2;
        gradient[3] += grad_beta_3;
        gradient[4] += grad_log_sigma;

        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}