#include <vulcan/light_tracker.h>
#include <vulcan/frame.h>

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

#include <fstream>
#include <sstream>

namespace vulcan
{

LightTracker::LightTracker() :
  depth_threshold_(0.2f),
  frame_(0),
  write_(false)
{
}

LightTracker::~LightTracker()
{
}

const Light& LightTracker::GetLight() const
{
  return light_;
}

void LightTracker::SetLight(const Light& light)
{
  light_ = light;
}

void LightTracker::BeginSolve(const Frame& frame)
{
  Tracker::BeginSolve(frame);
  ComputeKeyframeIntensities();
  ComputeFrameIntensities(frame);
  ComputeFrameGradients(frame);
  ComputeFrameMask(frame);
}

int LightTracker::GetResidualCount(const Frame& frame) const
{
  const int w = keyframe_->depth_image->GetWidth();
  const int h = keyframe_->depth_image->GetHeight();
  return w * h;
}

void LightTracker::ApplyUpdate(Frame& frame, Eigen::VectorXf& x) const
{
  Eigen::VectorXf update(6);
  update.setZero();

  for (int i = 0; i < x.size(); ++i)
  {
    update[i] = x[i];
  }

  Matrix4f Tinc;

  Tinc(0, 0) = 1.0;
  Tinc(0, 1) = -update[2];
  Tinc(0, 2) = +update[1];
  Tinc(0, 3) = +update[3];

  Tinc(1, 0) = +update[2];
  Tinc(1, 1) = 1.0;
  Tinc(1, 2) = -update[0];
  Tinc(1, 3) = +update[4];

  Tinc(2, 0) = -update[1];
  Tinc(2, 1) = +update[0];
  Tinc(2, 2) = 1.0;
  Tinc(2, 3) = +update[5];

  Tinc(3, 0) = 0.0;
  Tinc(3, 1) = 0.0;
  Tinc(3, 2) = 0.0;
  Tinc(3, 3) = 1.0;

  const Matrix4f M = Tinc * frame.Twc.GetInverseMatrix();

  Vector3f x_axis(M(0, 0), M(1, 0), M(2, 0));
  Vector3f y_axis(M(0, 1), M(1, 1), M(2, 1));
  Vector3f z_axis(M(0, 2), M(1, 2), M(2, 2));

  x_axis.Normalize();
  y_axis.Normalize();
  z_axis = x_axis.Cross(y_axis);
  y_axis = z_axis.Cross(x_axis);

  Matrix3f R;

  R(0, 0) = x_axis[0];
  R(1, 0) = x_axis[1];
  R(2, 0) = x_axis[2];

  R(0, 1) = y_axis[0];
  R(1, 1) = y_axis[1];
  R(2, 1) = y_axis[2];

  R(0, 2) = z_axis[0];
  R(1, 2) = z_axis[1];
  R(2, 2) = z_axis[2];

  Vector3f t;

  t[0] = M(0, 3);
  t[1] = M(1, 3);
  t[2] = M(2, 3);

  frame.Twc = (Transform::Translate(t) * Transform::Rotate(R)).Inverse();
}

void LightTracker::WriteDataFiles(const Frame& frame)
{
  ComputeResiduals(frame, residuals_);
  ComputeJacobian(frame, jacobian_);

  {
    std::vector<float> residuals(residuals_.GetSize());
    residuals_.CopyToHost(residuals.data());

    std::stringstream file;
    file << "residuals_";
    file << std::setw(2) << std::setfill('0') << frame_;
    file << ".txt";

    std::ofstream stream(file.str());

    for (float residual : residuals)
    {
      stream << residual << std::endl;
    }

    stream.close();
  }

  {
    std::vector<Vector6f> jacobian(jacobian_.GetSize());
    jacobian_.CopyToHost(jacobian.data());

    std::stringstream file;
    file << "jacobian_";
    file << std::setw(2) << std::setfill('0') << frame_;
    file << ".txt";

    std::ofstream stream(file.str());

    for (const Vector6f& J : jacobian)
    {
      for (int i = 0; i < 5; ++i)
      {
        stream << J[i] << " ";
      }

      stream << J[5] << std::endl;
    }

    stream.close();
  }

  frame_++;
}

} // namespace vulcan