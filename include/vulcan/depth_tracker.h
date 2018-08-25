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

    int GetResidualCount(const Frame& frame) const override;

    void ComputeSystem(const Frame& frame) override;
};

} // namespace vulcan