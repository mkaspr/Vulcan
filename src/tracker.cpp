#include <vulcan/tracker.h>
#include <Eigen/Cholesky>
#include <vulcan/exception.h>
#include <vulcan/frame.h>
#include <vulcan/volume.h>

namespace vulcan
{

Tracker::Tracker(std::shared_ptr<const Volume> volume) :
  max_iterations_(10),
  change_threshold_(1E-6),
  volume_(volume)
{
  Initialize();
}

Tracker::~Tracker()
{
  cublasDestroy(handle_);
}

std::shared_ptr<const Volume> Tracker::GetVolume() const
{
  return volume_;
}

std::shared_ptr<const Frame> Tracker::GetKeyframe() const
{
  return keyframe_;
}

void Tracker::SetKeyframe(std::shared_ptr<const Frame> keyframe)
{
  keyframe_ = keyframe;
}

void Tracker::Track(Frame& frame)
{
  BeginSolve();

  while (IsSolving())
  {
    UpdateSolve();
  }

  EndSolve();
}

bool Tracker::IsSolving() const
{
  return !HasHaulted() && !HasConverged();
}

bool Tracker::HasHaulted() const
{
  return iteration_ >= max_iterations_;
}

bool Tracker::HasConverged() const
{
  return GetCostChange() < GetMinCostChange();
}

double Tracker::GetCostChange() const
{
  return prev_cost_ - curr_cost_;
}

double Tracker::GetMinCostChange() const
{
  return change_threshold_ * prev_cost_;
}

void Tracker::BeginSolve()
{
  ValidateKeyframe();
  CreateState();

  // upload current intensity image
  // compute current image mipmaps
  // compute current image gradients of all mipmaps

  // first perform rotation-only solve
}

void Tracker::CreateState()
{
  // set initial frame pose with keyframe pose
  curr_cost_ = std::numeric_limits<double>::max();
  UpdateCost();
  iteration_ = 0;
}

void Tracker::ValidateKeyframe()
{
  VULCAN_ASSERT_MSG(keyframe_, "keyframe has not been set");
  VULCAN_ASSERT_MSG(keyframe_->depth_image, "keyframe missing depth image");
  VULCAN_ASSERT_MSG(keyframe_->color_image, "keyframe missing color image");
  VULCAN_ASSERT_MSG(keyframe_->normal_image, "keyframe missing normal image");
}

void Tracker::UpdateSolve()
{
  ComputeOperands();
  ComputeUpdate();
}

void Tracker::ComputeOperands()
{
  ComputeJacobian();
  ComputeHessian();
  ComputeGradient();
}

void Tracker::ComputeResidual()
{
}

void Tracker::ComputeJacobian()
{

}

void Tracker::ComputeUpdate()
{
  Eigen::LDLT<Eigen::Matrix<double, 6, 6>> solver;
  Eigen::Matrix<double, 6, 6> hessian;
  Eigen::Matrix<double, 6, 1> gradient;
  Eigen::Matrix<double, 6, 1> update;

  hessian_.CopyToHost(hessian.data());
  gradient_.CopyToHost(gradient.data());
  solver.compute(hessian);

  VULCAN_DEBUG(solver.info() == Eigen::Success);

  update = solver.solve(gradient);
  // state -= update; // TODO: convert back from tangent space
}

void Tracker::UpdateState()
{
  UpdateCost();
  ++iteration_;
}

void Tracker::EndSolve()
{
  // set final pose
  // check if "good" solution
}

void Tracker::Initialize()
{
  CreateHandle();
  CreateResidualBuffer();
  CreateJacobianBuffer();
  CreateHessianBuffer();
  CreateGradientBuffer();
}

void Tracker::CreateHandle()
{
  CUBLAS_ASSERT(cublasCreate(&handle_));
}

void Tracker::CreateResidualBuffer()
{
  // TODO: actually resize with image dimensions
  residuals_.Resize(640 * 480 * 3);
}

void Tracker::CreateJacobianBuffer()
{
  // TODO: actually resize with image dimensions
  residuals_.Resize(640 * 480 * 3 * 6);
}

void Tracker::CreateHessianBuffer()
{
  hessian_.Resize(36);
}

void Tracker::CreateGradientBuffer()
{
  gradient_.Resize(6);
}

} // namespace vulcan