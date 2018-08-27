#include <vulcan/depth_tracker.h>
#include <vulcan/frame.h>

namespace vulcan
{

DepthTracker::DepthTracker()
{
}

DepthTracker::~DepthTracker()
{
}

int DepthTracker::GetResidualCount(const Frame& frame) const
{
  const int w = frame.depth_image->GetWidth();
  const int h = frame.depth_image->GetHeight();
  return w * h;
}

void DepthTracker::ApplyUpdate(Frame& frame, Eigen::VectorXf& x) const
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
  Tinc(1, 2) = +update[0];
  Tinc(1, 3) = +update[4];

  Tinc(2, 0) = -update[1];
  Tinc(2, 1) = +update[0];
  Tinc(2, 2) = 1.0;
  Tinc(2, 3) = +update[5];

  Tinc(3, 0) = 0.0;
  Tinc(3, 1) = 0.0;
  Tinc(3, 2) = 0.0;
  Tinc(3, 3) = 1.0;

  const Matrix4f M = Tinc * frame.Twc.GetMatrix();

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

  frame.Twc = Transform::Translate(t) * Transform::Rotate(R);
}

} // namespace vulcan