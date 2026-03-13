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
        // Parameter layout (unconstrained):
        // position[0..4] = beta (unconstrained shape [4])
        // position[4] = sigma_log__ (LogTransform of sigma)
        
        let beta0 = position[0];
        let beta1 = position[1]; 
        let beta2 = position[2];
        let beta3 = position[3];
        let log_sigma = position[4];
        let sigma = log_sigma.exp();

        let mut logp = 0.0;

        // Zero gradients
        for i in 0..N_PARAMS {
            gradient[i] = 0.0;
        }

        // Prior: beta ~ Flat (contributes 0 to logp and gradient)
        // Flat priors have no contribution

        // Prior: sigma ~ HalfFlat with LogTransform
        // The Jacobian adjustment for LogTransform is +log_sigma
        logp += log_sigma;
        gradient[4] += 1.0; // d(log_sigma)/d(log_sigma) = 1

        // Likelihood: kid_score ~ Normal(mu, sigma)
        // From PyMC model:
        // mu = beta[0] + beta[1] * c2_mom_hs + beta[2] * c2_mom_iq + beta[3] * inter
        
        let inv_sigma = 1.0 / sigma;
        let inv_sigma_sq = inv_sigma * inv_sigma;
        let log_norm = -0.5 * LN_2PI - log_sigma;
        
        // Precompute gradient accumulators
        let mut grad_beta0 = 0.0;
        let mut grad_beta1 = 0.0; 
        let mut grad_beta2 = 0.0;
        let mut grad_beta3 = 0.0;
        let mut sum_residual_sq = 0.0;

        for i in 0..KID_SCORE_N {
            let y = KID_SCORE_DATA[i];
            
            // Covariate assignments (verified from previous validation):
            let c2_mom_hs = X_2_DATA[i];     // mom_hs - 0.5  
            let c2_mom_iq = X_1_DATA[i];     // mom_iq - 100.0
            let inter = X_0_DATA[i];         // c2_mom_hs * c2_mom_iq
            
            let mu = beta0 + beta1 * c2_mom_hs + beta2 * c2_mom_iq + beta3 * inter;
            let residual = y - mu;
            let residual_sq = residual * residual;
            let residual_scaled = residual * inv_sigma_sq;
            
            // Log-likelihood contribution
            logp += log_norm - 0.5 * residual_sq * inv_sigma_sq;
            
            // Gradient contributions for beta parameters
            grad_beta0 += residual_scaled;
            grad_beta1 += residual_scaled * c2_mom_hs;
            grad_beta2 += residual_scaled * c2_mom_iq;
            grad_beta3 += residual_scaled * inter;
            
            // Accumulate for sigma gradient
            sum_residual_sq += residual_sq;
        }

        // Gradient for log_sigma from Normal likelihood:
        // d/d(log_sigma) [-N*log(sigma) - 0.5*sum((y-mu)^2)/sigma^2]
        // = -N + sum((y-mu)^2)/sigma^2
        let n = KID_SCORE_N as f64;
        let grad_log_sigma_likelihood = -n + sum_residual_sq * inv_sigma_sq;

        // Assign gradients
        gradient[0] += grad_beta0;
        gradient[1] += grad_beta1;
        gradient[2] += grad_beta2;
        gradient[3] += grad_beta3;
        gradient[4] += grad_log_sigma_likelihood; // Already added the +1 from Jacobian above

        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}