#pragma once
// ABI boundary: no HYPRE types cross between the FP64 and FP32 libraries.
extern "C" {
void *bspf_amg32_create(int n, int nnz, const int *row, const int *col, const double *values);
int bspf_amg32_apply(void *handle, const double *device_rhs, double *device_solution);
void bspf_amg32_destroy(void *handle);
const char *bspf_amg32_error();
int bspf_amg32_scalar_bytes();
}
