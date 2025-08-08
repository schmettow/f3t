library(microbenchmark)
library(f3t)
test_that("FFT magnitude detects correct frequencies", {
  fs <- 4096  # sampling rate
  t <- seq(0, 1, length.out = fs)
  freqs <- c(50, 150, 300)  # known test frequencies
  signal <- generate_sine_mixture(t, freqs)

  # Call Rust FFT function (via savvy)
  spectrum <- fft_magnitude(signal)

  # Frequency axis
  freq_axis <- seq(0, fs - 1) * fs / length(spectrum)

  # Find top 3 frequency bins by magnitude
  top_bins <- order(spectrum, decreasing = TRUE)[1:3]
  detected_freqs <- sort(freq_axis[top_bins])

  # Sort target freqs to compare
  expected_freqs <- sort(freqs)

  # Allow for small error due to bin resolution
  expect_equal(detected_freqs, expected_freqs, tolerance = 5.0)
})


test_that("FFT magnitude detects correct frequencies", {
  fs <- 1024 * 1024  # sampling rate
  t <- seq(0, 1, length.out = fs)
  freqs <- c(5, 50, 150)  # known test frequencies
  signal <- generate_sine_mixture(t, freqs)

  # Call Rust FFT function (via savvy)
  spectrum <- fft_parallel(signal)

  # Frequency axis
  freq_axis <- seq(0, fs - 1) * fs / length(spectrum)

  # Find top 3 frequency bins by magnitude
  top_bins <- order(spectrum, decreasing = TRUE)[1:3]
  detected_freqs <- sort(freq_axis[top_bins])

  # Sort target freqs to compare
  expected_freqs <- sort(freqs)

  # Allow for small error due to bin resolution
  expect_equal(detected_freqs, expected_freqs, tolerance = 5.0)
})


test_that("fft_magnitude_sliding returns expected shape and content", {
  method <- fft_magnitude_sliding_parallel
  # --- Create test signal ---
  fs <- 48000
  duration <- 1
  t <- seq(0, duration, length.out = fs * duration)

  # Mixture of sine waves
  freqs <- c(440, 880, 1760)  # Musical tones: A4, A5, A6
  signal <- rowSums(sapply(freqs, function(f) sin(2 * pi * f * t)))
  plot(signal)


  # --- Run sliding FFT ---
  result <- method(signal)

  # --- Expectations ---
  window_size <- 1024
  hop_size <- window_size
  expected_windows <- floor((length(signal) - window_size) / hop_size) + 1

  expect_type(result, "double")
  expect_length(result, expected_windows * window_size)

  # --- Reshape into spectrogram ---
  spectrogram <- matrix(result, nrow = window_size, byrow = TRUE)
  expect_equal(dim(spectrogram), c(window_size, expected_windows))

  # --- Check for non-zero magnitudes ---
  expect_gt(mean(result), 0)

  # --- Optional: Performance check ---
  bench <- microbenchmark(method(signal), times = 3)
  print(bench)
  expect_lt(median(bench$time) / 1e9, 1.0)  # < 1 second runtime expected
  image(t(spectrogram), col = viridis::viridis(100), xlab = "Time", ylab = "Freq Bin")
})


# test_that("wavelet_cwt returns correct-sized output and runs successfully", {
#   # --- Generate test signal ---
#   fs <- 48000
#   duration <- 1
#   t <- seq(0, duration, length.out = fs * duration)
#
#   # Combine multiple sine waves
#   freqs <- c(440, 880, 1760)  # A4, A5, A6
#   signal <- rowSums(sapply(freqs, function(f) sin(2 * pi * f * t)))
#
#   # --- Run CWT ---
#   result <- wavelet_cwt(signal)
#
#   # --- Expectations ---
#   nscales <- 128
#   nsamples <- length(signal)
#
#   expect_type(result, "double")
#   expect_length(result, nscales * nsamples)
#
#   # Optional: check reshape works
#   mat <- matrix(result, nrow = nscales, byrow = TRUE)
#   expect_equal(dim(mat), c(nscales, nsamples))
#
#   # Optional: check for non-zero output
#   expect_gt(mean(result), 0)
#
#   # Optional: benchmark vs. threshold (1 second max)
#   bench <- microbenchmark(wavelet_cwt(signal), times = 3)
#   print(bench)
#   expect_lt(median(bench$time) / 1e9, 1.0)  # Less than 1 second
# })

library(microbenchmark)

test_that("Parallel sliding window FFT is faster than non-parallel version", {
  fs <- 4096  # large sample rate for meaningful load
  t_max <- 4
  t <- seq(0, t_max, length.out = fs * t_max)
  signal <- generate_sine_mixture(t, freqs = c(50, 150, 300))

  # Run benchmark
  bench <- microbenchmark(
    non_parallel = fft_magnitude_sliding(signal),
    parallel     = fft_magnitude_sliding_parallel(signal),
    times = 10L
  )

  print(bench)

  # Extract median timings (in nanoseconds)
  median_non_parallel <- median(bench$time[bench$expr == "non_parallel"])
  median_parallel     <- median(bench$time[bench$expr == "parallel"])

  # Test that the parallel version is faster
  expect_lt(median_parallel, median_non_parallel)
})


