#include <vulcan/tracker.h>
#include <Eigen/Cholesky>
#include <vulcan/exception.h>
#include <vulcan/frame.h>
#include <vulcan/volume.h>

namespace vulcan
{

Tracker::Tracker() :
  translation_enabled_(true),
  max_iterations_(20)
{
  Initialize();
}

Tracker::~Tracker()
{
  cublasDestroy(handle_);
}

std::shared_ptr<const Frame> Tracker::GetKeyframe() const
{
  return keyframe_;
}

void Tracker::SetKeyframe(std::shared_ptr<const Frame> keyframe)
{
  keyframe_ = keyframe;
}

bool Tracker::GetTranslationEnabled() const
{
  return translation_enabled_;
}

void Tracker::SetTranslationEnabled(bool enabled)
{
  translation_enabled_ = enabled;
}

int Tracker::GetMaxIterations() const
{
  return max_iterations_;
}

void Tracker::SetMaxIterations(int iterations)
{
  VULCAN_DEBUG(iterations > 0);
  max_iterations_ = iterations;
}

void Tracker::Track(Frame& frame)
{
  BeginSolve(frame);

  while (IsSolving())
  {
    UpdateSolve(frame);
  }

  EndSolve();
}

bool Tracker::IsSolving() const
{
  return iteration_ < max_iterations_;
}

void Tracker::BeginSolve(const Frame& frame)
{
  ValidateKeyframe();
  ValidateFrame(frame);
  ResizeBuffers(frame);
  CreateState(frame);
}

void Tracker::ValidateKeyframe() const
{
  VULCAN_DEBUG_MSG(keyframe_, "keyframe has not been assigned");
  VULCAN_DEBUG_MSG(keyframe_->depth_image, "keyframe missing depth image");
  VULCAN_DEBUG_MSG(keyframe_->normal_image, "keyframe missing normal image");

  const int dw = keyframe_->depth_image->GetWidth();
  const int dh = keyframe_->depth_image->GetHeight();
  const int nw = keyframe_->normal_image->GetWidth();
  const int nh = keyframe_->normal_image->GetHeight();

  VULCAN_DEBUG_MSG(dw > 0 && dh > 0, "invalid keyframe depth image size");
  VULCAN_DEBUG_MSG(dw == nw && dh == nh, "keyframe image size mismatch");
}

void Tracker::ValidateFrame(const Frame& frame) const
{
  VULCAN_DEBUG_MSG(frame.depth_image, "frame missing depth image");
  VULCAN_DEBUG_MSG(frame.normal_image, "frame missing normal image");

  const int dw = frame.depth_image->GetWidth();
  const int dh = frame.depth_image->GetHeight();
  const int nw = frame.normal_image->GetWidth();
  const int nh = frame.normal_image->GetHeight();

  VULCAN_DEBUG_MSG(dw > 0 && dh > 0, "invalid frame depth image size");
  VULCAN_DEBUG_MSG(dw == nw && dh == nh, "frame image size mismatch");
}

void Tracker::CreateState(const Frame& frame)
{
  iteration_ = 0;
}

void Tracker::UpdateSolve(Frame& frame)
{
  ComputeOperands(frame);
  ComputeUpdate(frame);
  UpdateState(frame);
}

void Tracker::ComputeOperands(const Frame& frame)
{
  ComputeSystem(frame);
}

void Tracker::ComputeUpdate(Frame& frame)
{
  const int parameter_count = GetParameterCount();
  Eigen::LDLT<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> solver;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> hessian;
  Eigen::Matrix<float, Eigen::Dynamic, 1> gradient;
  Eigen::Matrix<float, Eigen::Dynamic, 1> update;

  hessian.resize(parameter_count, parameter_count);
  gradient.resize(parameter_count);
  update.resize(parameter_count);

  hessian_.CopyToHost(hessian.data());
  gradient_.CopyToHost(gradient.data());

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> hessian2;
  hessian2.resize(parameter_count, parameter_count);

  int index = 0;

  for (int i = 0; i < parameter_count; ++i)
  {
    for (int j = 0; j <= i; ++j)
    {
      hessian2(i, j) = hessian.data()[index++];
    }
  }

  hessian2.triangularView<Eigen::StrictlyUpper>() = hessian2.transpose();
  solver.compute(hessian2);

  // std::cout << "Hessian:" << std::endl << hessian2 << std::endl << std::endl;

  VULCAN_DEBUG(solver.info() == Eigen::Success);

  update = -solver.solve(gradient);
  ApplyUpdate(frame, update);

  if (update.norm() < 1E-6) iteration_ = max_iterations_;
}

void Tracker::UpdateState(const Frame& frame)
{
  ++iteration_;
}

void Tracker::EndSolve()
{
}

void Tracker::ResizeBuffers(const Frame& frame)
{
  ResizeResidualBuffer(frame);
  ResizeJacobianBuffer(frame);
  ResizeHessianBuffer();
  ResizeGradientBuffer();
}

void Tracker::ResizeResidualBuffer(const Frame& frame)
{
  residuals_.Resize(GetResidualCount(frame));
}

void Tracker::ResizeJacobianBuffer(const Frame& frame)
{
  const int residual_count = GetResidualCount(frame);
  const int parameter_count = GetParameterCount();
  jacobian_.Resize(residual_count * parameter_count);
}

void Tracker::ResizeHessianBuffer()
{
  const int parameter_count = GetParameterCount();
  hessian_.Resize(parameter_count * parameter_count);
}

void Tracker::ResizeGradientBuffer()
{
  gradient_.Resize(GetParameterCount());
}

int Tracker::GetParameterCount() const
{
  return translation_enabled_ ? 6 : 3;
}

void Tracker::Initialize()
{
  CUBLAS_ASSERT(cublasCreate(&handle_));
}

} // namespace vulcan