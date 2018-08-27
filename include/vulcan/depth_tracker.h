#pragma once

#include <vulcan/matrix.h>
#include <vulcan/tracker.h>

namespace vulcan
{

class DepthTracker : public Tracker
{
  public:

    DepthTracker();

    virtual ~DepthTracker();

    void ComputeResiduals(const Frame& frame, Buffer<float>& residuals) const;

    void ComputeJacobian(const Frame& frame, Buffer<Vector6f>& jacobian) const;

  protected:

    int GetResidualCount(const Frame& frame) const override;

    void ApplyUpdate(Frame& frame, Eigen::VectorXf& x) const override;

    void ComputeSystem(const Frame& frame) override;
};

} // namespace vulcan