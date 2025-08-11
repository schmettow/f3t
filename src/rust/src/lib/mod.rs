// Example functions

//use std::{collections::VecDeque, slice::Windows};

pub mod spectro {
    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;
    use rustfft::{num_complex::Complex, FftPlanner};

    pub fn fft_sliding_parallel_inner(
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
            .map_err(|e| {
                println!("Error creating threadpool of size {}: {}", threads, e);
                e
            })
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
}

mod r {
    use crate::spectro::fft_sliding_parallel_inner;
    use savvy::savvy_err;
    use savvy::{savvy, NumericScalar, OwnedListSexp, OwnedRealSexp, RealSexp, Result, Sexp};
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
}

#[cfg(test)]
mod tests {
    use super::spectro::fft_sliding_parallel_inner;

    #[test]
    fn test_fft_sliding_parallel_inner_basic() {
        // Create a simple test signal - a sine wave
        let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let window_size = 16;
        let hop_size = 8;
        let threads = 2;

        let result = fft_sliding_parallel_inner(&signal, window_size, hop_size, threads);

        // Check that we get the expected number of windows
        let expected_windows = (signal.len() - window_size) / hop_size + 1;
        assert_eq!(result.len(), expected_windows);

        // Check that each spectrum has the right length (half + 1 for real FFT)
        for spectrum in &result {
            assert_eq!(spectrum.len(), window_size / 2 + 1);
        }

        // Check that all values are non-negative (magnitudes)
        for spectrum in &result {
            for &value in spectrum {
                assert!(value >= 0.0, "FFT magnitude should be non-negative");
            }
        }
    }

    #[test]
    fn test_fft_sliding_parallel_inner_single_window() {
        let signal: Vec<f64> = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let window_size = 8;
        let hop_size = 8;
        let threads = 1;

        let result = fft_sliding_parallel_inner(&signal, window_size, hop_size, threads);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), window_size / 2 + 1);
    }

    #[test]
    fn test_fft_sliding_parallel_inner_dc_signal() {
        // DC signal (constant value)
        let signal: Vec<f64> = vec![1.0; 32];
        let window_size = 16;
        let hop_size = 8;
        let threads = 1;

        let result = fft_sliding_parallel_inner(&signal, window_size, hop_size, threads);

        // For a DC signal, the first bin (DC component) should be large,
        // and other bins should be small
        for spectrum in &result {
            assert!(spectrum[0] > spectrum[1], "DC component should dominate");
        }
    }

    #[test]
    fn test_fft_sliding_parallel_inner_different_thread_counts() {
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.2).cos()).collect();
        let window_size = 16;
        let hop_size = 4;

        // Test with different thread counts - results should be identical
        let result1 = fft_sliding_parallel_inner(&signal, window_size, hop_size, 1);
        let result2 = fft_sliding_parallel_inner(&signal, window_size, hop_size, 4);

        assert_eq!(result1.len(), result2.len());
        for (spec1, spec2) in result1.iter().zip(result2.iter()) {
            assert_eq!(spec1.len(), spec2.len());
            for (val1, val2) in spec1.iter().zip(spec2.iter()) {
                assert!(
                    (val1 - val2).abs() < 1e-10,
                    "Results should be identical regardless of thread count"
                );
            }
        }
    }

    #[test]
    fn test_fft_sliding_parallel_inner_equal_spectrum_lengths() {
        let signal: Vec<f64> = (0..80).map(|i| (i as f64 * 0.15).sin()).collect();
        let window_size = 20;
        let hop_size = 10;
        let threads = 2;

        let result = fft_sliding_parallel_inner(&signal, window_size, hop_size, threads);

        // Check that all spectra have equal length
        if !result.is_empty() {
            let expected_length = window_size / 2 + 1;
            for (i, spectrum) in result.iter().enumerate() {
                assert_eq!(
                    spectrum.len(),
                    expected_length,
                    "Spectrum {} has length {} but expected {}",
                    i,
                    spectrum.len(),
                    expected_length
                );
            }

            // Also verify all spectra have the same length as each other
            let first_length = result[0].len();
            for (i, spectrum) in result.iter().enumerate() {
                assert_eq!(
                    spectrum.len(),
                    first_length,
                    "Spectrum {} has length {} but spectrum 0 has length {}",
                    i,
                    spectrum.len(),
                    first_length
                );
            }
        }
    }

    #[test]
    fn test_fft_sliding_parallel_inner_performance() {
        // Create a large signal to make parallelization benefits visible
        let signal: Vec<f64> = (0..10000).map(|i| (i as f64 * 0.01).sin()).collect();
        let window_size = 256;
        let hop_size = 128;

        // Measure time with single thread
        let start_single = std::time::Instant::now();
        let _result_single = fft_sliding_parallel_inner(&signal, window_size, hop_size, 1);
        let duration_single = start_single.elapsed();

        // Measure time with four threads
        let start_four = std::time::Instant::now();
        let _result_four = fft_sliding_parallel_inner(&signal, window_size, hop_size, 4);
        let duration_four = start_four.elapsed();

        // Four threads should be faster than single thread
        assert!(
            duration_four < duration_single,
            "Four threads ({:?}) should be faster than single thread ({:?})",
            duration_four,
            duration_single
        );
    }
}

pub mod cmd {
    use crate::spectro::fft_sliding_parallel_inner;
    use std::env;
    use std::fs::File;
    use std::io::{self, BufRead, BufReader, BufWriter, Write};

    pub fn run() -> io::Result<()> {
        let args: Vec<String> = env::args().collect();

        // Parse command line arguments
        let mut input_file: Option<String> = None;
        let mut output_file: Option<String> = None;
        let mut window_size: usize = 1024;
        let mut hop_size: usize = 512;
        let mut threads: usize = 4;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--input" | "-i" => {
                    i += 1;
                    if i < args.len() {
                        input_file = Some(args[i].clone());
                    }
                }
                "--output" | "-o" => {
                    i += 1;
                    if i < args.len() {
                        output_file = Some(args[i].clone());
                    }
                }
                "--window-size" | "-w" => {
                    i += 1;
                    if i < args.len() {
                        window_size = args[i].parse().unwrap_or(1024);
                    }
                }
                "--hop-size" | "-h" => {
                    i += 1;
                    if i < args.len() {
                        hop_size = args[i].parse().unwrap_or(512);
                    }
                }
                "--threads" | "-t" => {
                    i += 1;
                    if i < args.len() {
                        threads = args[i].parse().unwrap_or(4);
                    }
                }
                "--help" => {
                    print_help();
                    return Ok(());
                }
                _ => {}
            }
            i += 1;
        }

        // Read input signal
        let signal = read_signal(input_file.as_deref())?;

        // Process signal
        let spectra = fft_sliding_parallel_inner(&signal, window_size, hop_size, threads);

        // Write output
        write_spectra(&spectra, output_file.as_deref())?;

        Ok(())
    }

    fn read_signal(input_file: Option<&str>) -> io::Result<Vec<f64>> {
        let mut signal = Vec::new();

        match input_file {
            Some(filename) => {
                let file = File::open(filename)?;
                let reader = BufReader::new(file);

                for line in reader.lines() {
                    let line = line?;
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        // Try to parse as multiple space-separated values first
                        for value_str in trimmed.split_whitespace() {
                            if let Ok(value) = value_str.parse::<f64>() {
                                signal.push(value);
                            }
                        }
                    }
                }
            }
            None => {
                let stdin = io::stdin();
                let reader = BufReader::new(stdin.lock());

                for line in reader.lines() {
                    let line = line?;
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        // Try to parse as multiple space-separated values first
                        for value_str in trimmed.split_whitespace() {
                            if let Ok(value) = value_str.parse::<f64>() {
                                signal.push(value);
                            }
                        }
                    }
                }
            }
        }

        Ok(signal)
    }

    fn write_spectra(spectra: &[Vec<f64>], output_file: Option<&str>) -> io::Result<()> {
        match output_file {
            Some(filename) => {
                let file = File::create(filename)?;
                let mut writer = BufWriter::new(file);

                for (window_idx, spectrum) in spectra.iter().enumerate() {
                    write!(writer, "Window {}: ", window_idx)?;
                    for (i, &value) in spectrum.iter().enumerate() {
                        if i > 0 {
                            write!(writer, " ")?;
                        }
                        write!(writer, "{:.6}", value)?;
                    }
                    writeln!(writer)?;
                }

                writer.flush()?;
            }
            None => {
                let stdout = io::stdout();
                let mut writer = BufWriter::new(stdout.lock());

                for (window_idx, spectrum) in spectra.iter().enumerate() {
                    write!(writer, "Window {}: ", window_idx)?;
                    for (i, &value) in spectrum.iter().enumerate() {
                        if i > 0 {
                            write!(writer, " ")?;
                        }
                        write!(writer, "{:.6}", value)?;
                    }
                    writeln!(writer)?;
                }

                writer.flush()?;
            }
        }

        Ok(())
    }

    fn print_help() {
        println!("FFT Sliding Window Spectrogram");
        println!("Usage: program [OPTIONS]");
        println!();
        println!("OPTIONS:");
        println!("  -i, --input <FILE>        Input file (default: stdin)");
        println!("  -o, --output <FILE>       Output file (default: stdout)");
        println!("  -w, --window-size <SIZE>  Window size (default: 1024)");
        println!("  -h, --hop-size <SIZE>     Hop size (default: 512)");
        println!("  -t, --threads <COUNT>     Number of threads (default: 4)");
        println!("      --help                Show this help message");
        println!();
        println!("Input format: One or more floating point numbers per line, space-separated");
        println!("Output format: One spectrum per line, space-separated magnitudes");
    }
}
