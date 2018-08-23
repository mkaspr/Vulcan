#include <vulcan/light_tracker.h>
#include <vulcan/frame.h>

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

namespace vulcan
{

LightTracker::LightTracker()
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
}

int LightTracker::GetResidualCount(const Frame& frame) const
{
  const int w = keyframe_->depth_image->GetWidth();
  const int h = keyframe_->depth_image->GetHeight();
  return w * h;
}

} // namespace vulcan