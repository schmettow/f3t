#' Generate a mixture of three sine waves
#'
#' @param t Vector of time points
#' @param freqs A numeric vector of 3 frequencies (Hz)
#' @param amps A numeric vector of 3 amplitudes (default = c(1, 1, 1))
#' @param phases A numeric vector of 3 phases (radians, default = c(0, 0, 0))
#' @return Numeric vector of the sine wave mixture
#' @export
r_generate_sine_mixture <- function(t, freqs, amps = c(1, 1, 1), phases = c(0, 0, 0)) {
  if (length(freqs) != 3 || length(amps) != 3 || length(phases) != 3) {
    stop("freqs, amps, and phases must be vectors of length 3")
  }

  signal <- amps[1] * sin(2 * pi * freqs[1] * t + phases[1]) +
    amps[2] * sin(2 * pi * freqs[2] * t + phases[2]) +
    amps[3] * sin(2 * pi * freqs[3] * t + phases[3])

  return(signal)
}

