#include <vulcan/tracker.h>
#include <Eigen/Cholesky>
#include <vulcan/exception.h>
#include <vulcan/frame.h>
#include <vulcan/volume.h>

namespace vulcan
{

inline Transform exp(const Eigen::VectorXd& omega)
{
  const double theta_sq = omega.head<3>().squaredNorm();
  const double theta = sqrt(theta_sq);
  const double half_theta = 0.5 * theta;

  double imag_factor;
  double real_factor;

  if(theta < 1E-10)
  {
    const double theta_po4 = theta_sq * theta_sq;
    imag_factor = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_po4;
    real_factor = 1 - 0.5 * theta_sq + (1.0 / 384.0) * theta_po4;
  }
  else
  {
    const double sin_half_theta = sin(half_theta);
    imag_factor = sin_half_theta / theta;
    real_factor = cos(half_theta);
  }

  Eigen::Vector3d xxx;
  xxx[0] = imag_factor * omega[0];
  xxx[1] = imag_factor * omega[1];
  xxx[2] = imag_factor * omega[2];

  const Transform R = Transform::Rotate(real_factor, xxx[0], xxx[1], xxx[2]);

  const Transform T = (omega.size() == 6) ?
      Transform::Translate(omega[3], omega[4], omega[5]) : Transform();

  return T * R;
}

Tracker::Tracker() :
  translation_enabled_(true),
  max_iterations_(5)
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

void Tracker::Track(Frame& frame)
{
  BeginSolve(frame);

  while (IsSolving())
  {
    UpdateSolve(frame);
  }

  EndSolve();

  // std::cout << "Transform:" << std::endl << frame.Tcw.GetMatrix() << std::endl;
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

  const int w = frame.depth_image->GetWidth();
  const int h = frame.depth_image->GetHeight();

  VULCAN_DEBUG_MSG(w > 0 && h > 0, "invalid frame depth image size");
}

void Tracker::CreateState(const Frame& frame)
{
  iteration_ = 0;
}

void Tracker::UpdateSolve(Frame& frame)
{
  // std::cout << "=======================" << std::endl;

  // Eigen::Matrix<double, Eigen::Dynamic, 1> r1(residuals_.GetSize());
  // Eigen::Matrix<double, Eigen::Dynamic, 1> r2(residuals_.GetSize());
  // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jacobian;
  // jacobian.resize(GetResidualCount(frame), GetParameterCount());

  // ComputeResidual(frame);
  // ComputeOperands(frame);
  // residuals_.CopyToHost(r1.data());
  // jacobian_.CopyToHost(jacobian.data());

  // Frame other_frame = frame;
  // const float step = 1E-6f;
  // other_frame.Tcw = Transform::Translate(0, 0, step) * other_frame.Tcw;
  // ComputeResidual(other_frame);
  // ComputeOperands(other_frame);
  // residuals_.CopyToHost(r2.data());

  // const int offset = 400 * 640 + 320;
  // std::cout << "analytical:" << std::endl << jacobian.block<10, 6>(offset, 0) << std::endl;
  // std::cout << std::endl;
  // std::cout << "finite diff:" << std::endl << (r2.block<10, 1>(offset, 0) - r1.block<10, 1>(offset, 0)) / step << std::endl;

  // std::cout << "=======================" << std::endl;

  ComputeOperands(frame);
  ComputeUpdate(frame);
  UpdateState(frame);
}

void Tracker::ComputeOperands(const Frame& frame)
{
  ComputeResidual(frame);
  ComputeJacobian(frame);
  ComputeHessian();
  ComputeGradient();
}

void Tracker::ComputeUpdate(Frame& frame)
{
  const int parameter_count = GetParameterCount();
  Eigen::LDLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> solver;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> hessian;
  Eigen::Matrix<double, Eigen::Dynamic, 1> gradient;
  Eigen::Matrix<double, Eigen::Dynamic, 1> update;

  hessian.resize(parameter_count, parameter_count);
  gradient.resize(parameter_count);
  update.resize(parameter_count);

  hessian_.CopyToHost(hessian.data());
  gradient_.CopyToHost(gradient.data());
  solver.compute(hessian);

  VULCAN_DEBUG(solver.info() == Eigen::Success);

  update = -solver.solve(gradient);
  frame.Tcw = exp(update).Inverse() * frame.Tcw;

  // frame.Tcw = exp(update) * frame.Tcw;
  // frame.Tcw = frame.Tcw * exp(update).Inverse();
  // frame.Tcw = frame.Tcw * exp(update);

  // Eigen::Matrix<double, Eigen::Dynamic, 1> residuals(residuals_.GetSize());
  // residuals_.CopyToHost(residuals.data());

  // std::cout << "residuals " << iteration_ << ":   " <<  residuals.block<10, 1>(400 * 640 + 320, 0).transpose() << std::endl;
  // std::cout << "gradient " << iteration_ << ":    " <<  gradient.transpose() << std::endl;
  // std::cout << "udpate " << iteration_ << ":      " <<  update.transpose() << std::endl;
  // std::cout << "translation " << iteration_ << ": " <<  frame.Tcw.GetTranslation().Transpose() << std::endl;
  // std::cout << "hessian:" << std::endl << hessian << std::endl;
  // std::cout << std::endl;
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

int Tracker::GetResidualCount(const Frame& frame) const
{
  const int w = frame.depth_image->GetWidth();
  const int h = frame.depth_image->GetHeight();
  return w * h;
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