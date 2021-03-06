#pragma once

#include <vulcan/tracker.h>
#include <vulcan/image.h>

namespace vulcan
{

class ColorTracker : public Tracker
{
  public:

    ColorTracker();

    virtual ~ColorTracker();

    void ComputeResiduals(const Frame& frame, Buffer<float>& residuals);

    void ComputeJacobian(const Frame& frame, Buffer<Vector6f>& jacobian);

  protected:

    void BeginSolve(const Frame& frame) override;

    int GetResidualCount(const Frame &frame) const override;

    void ApplyUpdate(Frame& frame, Eigen::VectorXf& x) const override;

    void ComputeSystem(const Frame& frame) override;

    void ComputeKeyframeIntensities();

    void ComputeFrameIntensities(const Frame& frame);

    void ComputeFrameGradients(const Frame& frame);

  protected:

    Image keyframe_intensities_;

    Image frame_intensities_;

    Image frame_gradient_x_;

    Image frame_gradient_y_;
};

} // namespace vulcan