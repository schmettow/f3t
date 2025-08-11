
#include <stdint.h>
#include <Rinternals.h>
#include <R_ext/Parse.h>

#include "rust/api.h"

static uintptr_t TAGGED_POINTER_MASK = (uintptr_t)1;

SEXP handle_result(SEXP res_) {
    uintptr_t res = (uintptr_t)res_;

    // An error is indicated by tag.
    if ((res & TAGGED_POINTER_MASK) == 1) {
        // Remove tag
        SEXP res_aligned = (SEXP)(res & ~TAGGED_POINTER_MASK);

        // Currently, there are two types of error cases:
        //
        //   1. Error from Rust code
        //   2. Error from R's C API, which is caught by R_UnwindProtect()
        //
        if (TYPEOF(res_aligned) == CHARSXP) {
            // In case 1, the result is an error message that can be passed to
            // Rf_errorcall() directly.
            Rf_errorcall(R_NilValue, "%s", CHAR(res_aligned));
        } else {
            // In case 2, the result is the token to restart the
            // cleanup process on R's side.
            R_ContinueUnwind(res_aligned);
        }
    }

    return (SEXP)res;
}

SEXP savvy_fft_sliding_parallel__impl(SEXP c_arg__signal, SEXP c_arg__window_size, SEXP c_arg__hop_size, SEXP c_arg__threads) {
    SEXP res = savvy_fft_sliding_parallel__ffi(c_arg__signal, c_arg__window_size, c_arg__hop_size, c_arg__threads);
    return handle_result(res);
}

SEXP savvy_generate_sine_mixture__impl(SEXP c_arg__duration, SEXP c_arg__sampling_rate, SEXP c_arg__freqs, SEXP c_arg__amps, SEXP c_arg__phases, SEXP c_arg__noise_sd) {
    SEXP res = savvy_generate_sine_mixture__ffi(c_arg__duration, c_arg__sampling_rate, c_arg__freqs, c_arg__amps, c_arg__phases, c_arg__noise_sd);
    return handle_result(res);
}


static const R_CallMethodDef CallEntries[] = {
    {"savvy_fft_sliding_parallel__impl", (DL_FUNC) &savvy_fft_sliding_parallel__impl, 4},
    {"savvy_generate_sine_mixture__impl", (DL_FUNC) &savvy_generate_sine_mixture__impl, 6},
    {NULL, NULL, 0}
};

void R_init_f3t(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);

    // Functions for initialzation, if any.

}
