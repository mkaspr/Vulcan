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

} // namespace vulcan