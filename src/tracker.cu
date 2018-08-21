#include <vulcan/tracker.h>
#include <vulcan/device.h>
#include <vulcan/frame.h>
#include <vulcan/projection.h>
#include <vulcan/transform.h>

namespace vulcan
{

void Tracker::ComputeHessian()
{
  const double one = 1;
  const double zero = 0;
  const int residual_count = residuals_.GetSize();
  const int parameter_count = GetParameterCount();
  const double* jacobian = jacobian_.GetData();
  double* hessian = hessian_.GetData();

  CUBLAS_DEBUG(cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, parameter_count,
      parameter_count, residual_count, &one, jacobian, residual_count,
      jacobian, residual_count, &zero, hessian, parameter_count));
}

void Tracker::ComputeGradient()
{
  const double one = 1;
  const double zero = 0;
  const int residual_count = residuals_.GetSize();
  const int parameter_count = GetParameterCount();
  const double* jacobian = jacobian_.GetData();
  const double* residuals = residuals_.GetData();
  double* gradient = gradient_.GetData();

  CUBLAS_DEBUG(cublasDgemv(handle_, CUBLAS_OP_T, residual_count,
    parameter_count, &one, jacobian, residual_count, residuals, 1, &zero,
    gradient, 1));
}

} // namespace vulcan