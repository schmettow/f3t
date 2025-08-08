# Fast Fast Fourier Transformation

The goal of this R package is to efficiently perform spectral analysis for very long signals. It provides a multi-core implementation of sliding window FFT using a Rust backend.

The Sliding-window Fast Fourier Transform (FFT) runs over a long time series in overlapping segments (windows). This technique provides spectral analysis where the temporal resolution is only limited to theoretical bounds.

# Installation

You can install the package from GitHub using the devtools package:

```{r}
devtools::install_github("schmettow/f3t")
```

# Usage

```{r}
library(f3t)

rate <- 512  # large sample rate for meaningful load
secs <- 3600 * 24 # one day
freqs = c(7, 41, 221) # mixture of three frequencies
ampls <- c(2, 3, 5)
phases <- c(0,0,0)
error = 19
signal <- f3t::generate_sine_mixture(secs, rate, freqs, ampls, phases, error)
spectra <- f3t::fft_sliding_parallel(signal, window_size, hop_size, threads = 4)
```
