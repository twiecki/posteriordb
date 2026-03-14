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

// Define ln(2*pi) constant
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
        // Extract unconstrained parameters
        let beta0 = position[0];
        let beta1 = position[1]; 
        let beta2 = position[2];
        let beta3 = position[3];
        let log_sigma = position[4];

        // Transform sigma from log space
        let sigma = log_sigma.exp();

        // Initialize logp and gradient
        let mut logp = 0.0;
        gradient.fill(0.0);

        // Prior: beta ~ Flat (improper uniform) - contributes 0 to logp and gradient
        // (already initialized to 0)

        // Prior: sigma ~ HalfCauchy(beta=2.5) with LogTransform
        // HalfCauchy(x | beta) = 2/(pi * beta * (1 + (x/beta)^2)) for x >= 0
        // With LogTransform: x = exp(log_x), Jacobian = exp(log_x) = x
        // logp = log(2) - log(pi) - log(beta) - log(1 + (x/beta)^2) + log_x
        let beta_cauchy = 2.5;
        let sigma_scaled = sigma / beta_cauchy;
        let cauchy_term = 1.0 + sigma_scaled * sigma_scaled;
        logp += (2.0_f64).ln() - std::f64::consts::PI.ln() - beta_cauchy.ln() - cauchy_term.ln() + log_sigma;
        
        // Gradient for log_sigma (HalfCauchy + LogTransform)
        // d/d(log_x) = -2*(x/beta)^2/(1 + (x/beta)^2) * (1/beta) * x + 1
        //            = -2*(x/beta)^3/(1 + (x/beta)^2) + 1
        let d_logp_d_log_sigma = -2.0 * sigma_scaled * sigma_scaled * sigma_scaled / (cauchy_term * beta_cauchy) + 1.0;
        gradient[4] += d_logp_d_log_sigma;

        // Likelihood: kid_score ~ Normal(mu, sigma)
        // mu = beta[0] + beta[1] * mom_hs + beta[2] * mom_iq + beta[3] * (mom_hs * mom_iq)
        // Based on the data description:
        // - X_2_DATA should be mom_hs (binary, range [0,1])
        // - X_0_DATA should be mom_iq (range [0, 138.893]) 
        // - X_1_DATA appears to be mom_iq centered (range [71.037, 138.893], mean=100)
        
        let inv_sigma = 1.0 / sigma;
        let inv_sigma_sq = inv_sigma * inv_sigma;
        let log_norm = -0.5 * LN_2PI - log_sigma;

        // Accumulate gradients efficiently
        let mut grad_beta0 = 0.0;
        let mut grad_beta1 = 0.0; 
        let mut grad_beta2 = 0.0;
        let mut grad_beta3 = 0.0;
        let mut grad_log_sigma_from_likelihood = 0.0;

        for i in 0..KID_SCORE_N {
            // Get predictor values for observation i
            let mom_iq = X_1_DATA[i];      // mom_iq (centered at 100)
            let mom_hs = X_2_DATA[i];      // mom_hs (0 or 1)
            let interaction = mom_hs * mom_iq; // mom_hs * mom_iq
            
            // Linear predictor: mu_i = beta0 + beta1*mom_hs + beta2*mom_iq + beta3*(mom_hs * mom_iq)
            let mu_i = beta0 + beta1 * mom_hs + beta2 * mom_iq + beta3 * interaction;
            let residual = KID_SCORE_DATA[i] - mu_i;
            
            // Normal log-likelihood for this observation
            logp += log_norm - 0.5 * residual * residual * inv_sigma_sq;
            
            // Gradient accumulation
            let scaled_residual = residual * inv_sigma_sq;
            grad_beta0 += scaled_residual;
            grad_beta1 += scaled_residual * mom_hs;
            grad_beta2 += scaled_residual * mom_iq;
            grad_beta3 += scaled_residual * interaction;
            
            // Gradient w.r.t. log_sigma from likelihood
            // d/d(log_sigma) [-0.5*log(2*pi) - log_sigma - 0.5*(y-mu)^2/sigma^2]
            // = -1 + (y-mu)^2/sigma^2
            grad_log_sigma_from_likelihood += residual * residual * inv_sigma_sq - 1.0;
        }

        // Add gradients to output
        gradient[0] += grad_beta0;
        gradient[1] += grad_beta1;
        gradient[2] += grad_beta2;
        gradient[3] += grad_beta3;
        gradient[4] += grad_log_sigma_from_likelihood;

        Ok(logp)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
        &mut self, _rng: &mut R, array: &[f64],
    ) -> Result<Draw, CpuMathError> {
        Ok(Draw { parameters: array.to_vec() })
    }
}