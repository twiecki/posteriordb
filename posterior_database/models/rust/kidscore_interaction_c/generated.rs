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
        // Parameters
        let beta = [position[0], position[1], position[2], position[3]]; // beta[4] - Flat prior
        let log_sigma = position[4]; // log-transformed sigma
        let sigma = log_sigma.exp();
        
        let mut logp = 0.0;
        
        // Initialize gradients
        gradient.fill(0.0);
        
        // ═══ PRIORS ═══
        
        // beta ~ Flat() - contributes 0 to logp and gradient
        // (Flat prior has infinite support, logp = 0 everywhere)
        
        // sigma ~ HalfFlat() with LogTransform 
        // HalfFlat has logp = 0 for sigma > 0, -inf for sigma <= 0
        // But since sigma = exp(log_sigma), sigma is always > 0
        // Only the Jacobian +log_sigma contributes
        logp += log_sigma;
        gradient[4] += 1.0; // d/d(log_sigma) [log_sigma] = 1
        
        // ═══ LIKELIHOOD ═══
        
        // Precompute common terms for efficiency
        let inv_sigma = 1.0 / sigma;
        let inv_sigma_sq = inv_sigma * inv_sigma;
        let log_norm_term = -0.5 * LN_2PI - log_sigma;
        
        // Gradient accumulators for vectorization
        let mut grad_beta = [0.0f64; 4];
        let mut grad_log_sigma = 0.0f64;
        
        // kid_score ~ Normal(mu, sigma) for N=434 observations
        // mu = beta[0] + beta[1] * c_mom_hs + beta[2] * c_mom_iq + beta[3] * interaction
        // Final mapping based on gradient validation:
        // X_0_DATA = interaction, X_1_DATA = c_mom_iq, X_2_DATA = c_mom_hs
        for i in 0..KID_SCORE_N {
            let y = KID_SCORE_DATA[i];
            let interaction = X_0_DATA[i];   // X_0 = interaction  
            let c_mom_iq = X_1_DATA[i];      // X_1 = c_mom_iq 
            let c_mom_hs = X_2_DATA[i];      // X_2 = c_mom_hs
            
            let mu = beta[0] + beta[1] * c_mom_hs + beta[2] * c_mom_iq + beta[3] * interaction;
            let residual = y - mu;
            let residual_scaled = residual * inv_sigma;
            
            // Log-likelihood: -0.5*ln(2π) - ln(σ) - 0.5*((y - μ)/σ)²
            logp += log_norm_term - 0.5 * residual_scaled * residual_scaled;
            
            // Gradients
            let common_factor = residual * inv_sigma_sq;
            
            // d(logp)/d(beta_j) = (y - mu) / sigma² * (-d(mu)/d(beta_j))
            grad_beta[0] += common_factor;                    // d(mu)/d(beta[0]) = 1
            grad_beta[1] += common_factor * c_mom_hs;        // d(mu)/d(beta[1]) = c_mom_hs
            grad_beta[2] += common_factor * c_mom_iq;        // d(mu)/d(beta[2]) = c_mom_iq
            grad_beta[3] += common_factor * interaction;     // d(mu)/d(beta[3]) = interaction
            
            // d(logp)/d(log_sigma) from Normal likelihood
            // Normal contributes: -log_sigma - 0.5 * residual²/sigma²
            // d/d(log_sigma) [-log_sigma] = -1
            // d/d(log_sigma) [-0.5 * residual²/sigma²] = residual²/sigma² (since d(sigma²)/d(log_sigma) = 2σ²)
            grad_log_sigma += residual_scaled * residual_scaled - 1.0;
        }
        
        // Apply gradient accumulators
        gradient[0] += grad_beta[0];
        gradient[1] += grad_beta[1];
        gradient[2] += grad_beta[2];
        gradient[3] += grad_beta[3];
        gradient[4] += grad_log_sigma;
        
        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}