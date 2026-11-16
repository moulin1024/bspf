// MFEM 4.9's generic CUDA Values3D fallback has MAX_INTERP_1D=8.
// Q2 geometry sampled at p=8 GLL vertices needs D1D=3, Q1D=9.
// Register the existing MFEM kernel with correctly sized static storage.
#include "mfem.hpp"
#include "fem/qinterp/eval.hpp"
void RegisterLORGeometryKernel()
{
  mfem::QuadratureInterpolator::TensorEvalKernels::
    Specialization<3,mfem::QVectorLayout::byVDIM,3,3,9>::Opt<1>::Add();
}
