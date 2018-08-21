#include <vulcan/color_tracker.h>
#include <vulcan/frame.h>

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

namespace vulcan
{

ColorTracker::ColorTracker()
{
}

ColorTracker::~ColorTracker()
{
}

void ColorTracker::BeginSolve(const Frame& frame)
{
  Tracker::BeginSolve(frame);
  ComputeKeyframeIntensities();
  ComputeFrameIntensities(frame);
  ComputeFrameGradients(frame);
}

int ColorTracker::GetResidualCount(const Frame& frame) const
{
  const int w = keyframe_->depth_image->GetWidth();
  const int h = keyframe_->depth_image->GetHeight();
  return w * h;
}

} // namespace vulcan