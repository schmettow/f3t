// Example functions

//use std::{collections::VecDeque, slice::Windows};

use ndrustfft::nddct1_par;
use rustfft::{num_complex::Complex, FftPlanner};
use savvy::{savvy, NumericScalar, OwnedListSexp, OwnedRealSexp, RealSexp, Result, Sexp};

/// Compute FFT magnitudes of a real-valued input slice.
/// Returns a Vec<f64> of magnitudes (length = input.len()).
/// #@export
#[savvy]
pub fn fft_magnitude(input: RealSexp) -> Result<Sexp> {
    let len = input.len();
    // Convert real input to complex
    let mut buffer: Vec<Complex<f64>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();

    // Plan and run FFT
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(len);
    fft.process(&mut buffer);

    // Compute magnitude for each FFT bin
    let out: Vec<f64> = buffer.iter().map(|c| c.norm()).collect();
    let out: OwnedRealSexp = out.try_into()?;
    Result::Ok(out.into())
}

use ndarray::Array1;
/// Compute FFT magnitudes of a real-valued input slice.
/// Returns a Vec<f64> of magnitudes (length = input.len()).
/// # @export
#[savvy]
pub fn fft_parallel(input: RealSexp) -> Result<Sexp> {
    let len = input.len();
    // Convert real input to complex
    //let buffer: Vec<Complex<f64>> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    //let mut output: Vec<Complex<f64>> = vec![Complex::default(); len];
    let input: Vec<f64> = input.to_vec();
    let input = Array1::<f64>::from(input);
    let mut output = Array1::<f64>::zeros(len);

    // Plan and run FFT
    let mut planner = ndrustfft::DctHandler::<f64>::new(len);
    nddct1_par(&input, &mut output, &mut planner, 0);
    let output = output.to_vec();
    Result::Ok(output.try_into()?)
}

/// Compute FFT magnitudes of a real-valued input slice.
/// Returns a Vec<f64> of magnitudes (length = input.len()).
/// #@export
#[savvy]
pub fn fft_magnitude_sliding(input: RealSexp) -> Result<Sexp> {
    let signal: Vec<f64> = input.iter().cloned().collect();
    let window_size = 1024; // Fixed size; you could expose this as an argument.
    let hop_size = window_size; // No overlap; use <window_size for overlap

    let len = signal.len();
    let n_windows = (len - window_size) / hop_size + 1;

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(window_size);

    let mut output = Vec::with_capacity(n_windows * window_size);

    for i in 0..n_windows {
        let start = i * hop_size;
        let end = start + window_size;
        let window = &signal[start..end];

        // Convert real window to complex
        let mut buffer: Vec<Complex<f64>> = window.iter().map(|&x| Complex::new(x, 0.0)).collect();

        // Run FFT
        fft.process(&mut buffer);

        // Compute magnitudes
        let magnitudes: Vec<f64> = buffer.iter().map(|c| c.norm()).collect();

        // Append to output (row-wise)
        output.extend(magnitudes);
    }

    let out: OwnedRealSexp = output.try_into()?;
    Ok(out.into())
}

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use savvy::savvy_err;

fn fft_sliding_parallel_inner(
    signal: &[f64],
    window_size: usize,
    hop_size: usize,
    threads: usize,
) -> Vec<Vec<f64>> {
    let n_windows = (signal.len() - window_size) / hop_size + 1;

    // Precompute FFT plan (shared across threads)
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(window_size);
    let fft = &*Box::leak(Box::new(fft)); // Safe for threads; single shared reference

    // Build and use a thread pool with the specified number of threads
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads as usize)
        .build()
        .map_err(|e| savvy_err!("Error creating threadpool of size {}: {}", threads, e))
        .unwrap();
    let spectra: Vec<Vec<f64>> = pool.install(|| {
        (0..n_windows)
            .into_par_iter()
            .map(|i| {
                let start = i * hop_size;
                let end = start + window_size;
                let window = &signal[start..end];
                let mut buffer: Vec<Complex<f64>> =
                    window.iter().map(|&x| Complex::new(x, 0.0)).collect();

                fft.process(&mut buffer);

                buffer
                    .iter()
                    .take(window_size / 2 + 1)
                    .map(|c| c.norm())
                    .collect()
            })
            .collect()
    });

    spectra
}

/// Parallel FFT over all windows
/// @export
#[savvy]
pub fn fft_sliding_parallel(
    signal: RealSexp,
    window_size: NumericScalar,
    hop_size: NumericScalar,
    threads: NumericScalar,
) -> Result<Sexp> {
    let signal: Vec<f64> = signal.to_vec();
    let window_size = window_size.as_usize()?;
    let hop_size = hop_size.as_usize()?;
    let len: usize = signal.len();
    let n_windows = (len - window_size) / hop_size + 1;
    let threads = threads.as_usize()?;
    let spectra = fft_sliding_parallel_inner(&signal, window_size, hop_size, threads);

    // Convert Vec<Vec<f64>> to ListSexp (R list of numeric vectors)
    let mut list = OwnedListSexp::new(n_windows, true)?;
    for (i, spectrum) in spectra.iter().enumerate() {
        let spectrum_sexp: OwnedRealSexp = spectrum.as_slice().try_into()?;
        list.set_name(i, format!("{}", i + 1).as_str())?;
        list.set_value(i, spectrum_sexp)?;
    }
    Ok(list.into())
}

use rand::rng;
use rand_distr::{Distribution, Normal};

/// Parallel FFT over all windows
/// @export
#[savvy]
pub fn generate_sine_mixture(
    duration: NumericScalar,
    sampling_rate: NumericScalar,
    freqs: RealSexp,
    amps: RealSexp,
    phases: RealSexp,
    noise_sd: NumericScalar,
) -> savvy::Result<savvy::Sexp> {
    let freqs: Vec<f64> = freqs.iter().cloned().collect();
    let amps: Vec<f64> = amps.iter().cloned().collect();
    let phases: Vec<f64> = phases.iter().cloned().collect();

    // Validate input lengths
    if freqs.len() != 3 || amps.len() != 3 || phases.len() != 3 {
        savvy_err!("Frequencies, amplitudes, and phases must be vectors of length 3");
    }

    // Compute the sine mixture
    let two_pi = std::f64::consts::PI * 2.0;

    let n_samples = (duration.as_f64() * sampling_rate.as_f64()).round() as usize;
    let dt = 1.0 / sampling_rate.as_f64();

    let noise = Normal::new(0.0, noise_sd.as_f64())
        .map_err(|_| savvy_err!("Invalid noise_std (must be >= 0)"))?;
    let mut rng = rng();

    let signal: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 * dt;
            let sine_sum = amps[0] * (two_pi * freqs[0] * t + phases[0]).sin()
                + amps[1] * (two_pi * freqs[1] * t + phases[1]).sin()
                + amps[2] * (two_pi * freqs[2] * t + phases[2]).sin();
            let noise = if noise_sd.as_f64() > 0.0 {
                noise.sample(&mut rng)
            } else {
                0.0
            };
            sine_sum + noise
        })
        .collect();

    // Return as R vector
    let output: OwnedRealSexp = signal.try_into()?;
    Ok(output.into())
}
