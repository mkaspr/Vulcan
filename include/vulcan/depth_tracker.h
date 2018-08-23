#pragma once

#include <vulcan/tracker.h>

namespace vulcan
{

class DepthTracker : public Tracker
{
  public:

    DepthTracker();

    virtual ~DepthTracker();

  protected:

    void ComputeResidual(const Frame& frame) override;

    void ComputeJacobian(const Frame& frame) override;

    int GetResidualCount(const Frame& frame) const override;

    void ComputeSystem(const Frame& frame) override;
};

} // namespace vulcan