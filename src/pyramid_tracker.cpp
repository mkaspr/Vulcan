#include <vulcan/pyramid_tracker.h>
#include <vulcan/frame.h>
#include <vulcan/color_tracker.h>
#include <vulcan/depth_tracker.h>
#include <vulcan/light_tracker.h>

namespace vulcan
{

template<typename Tracker>
PyramidTracker<Tracker>::PyramidTracker() :
  tracker_(std::shared_ptr<Tracker>(new Tracker())),
  half_keyframe_(std::make_shared<Frame>()),
  quarter_keyframe_(std::make_shared<Frame>())
{
}

template<typename Tracker>
PyramidTracker<Tracker>::PyramidTracker(std::shared_ptr<Tracker> tracker) :
  tracker_(tracker),
  half_keyframe_(std::make_shared<Frame>()),
  quarter_keyframe_(std::make_shared<Frame>())
{
}

template<typename Tracker>
PyramidTracker<Tracker>::~PyramidTracker()
{
}

template<typename Tracker>
std::shared_ptr<const Tracker> PyramidTracker<Tracker>::GetTracker() const
{
  return tracker_;
}

template<typename Tracker>
std::shared_ptr<const Frame> PyramidTracker<Tracker>::GetKeyframe() const
{
  return keyframe_;
}

template<typename Tracker>
void PyramidTracker<Tracker>::SetKeyframe(std::shared_ptr<const Frame> keyframe)
{
  keyframe_ = keyframe;
}

template<typename Tracker>
void PyramidTracker<Tracker>::Track(Frame& frame)
{
  VULCAN_DEBUG(keyframe_);

  Frame half_frame;
  Frame quarter_frame;

  frame.Downsample(half_frame);
  half_frame.Downsample(quarter_frame);

  frame.Downsample(*half_keyframe_);
  half_keyframe_->Downsample(*quarter_keyframe_);

  tracker_->SetMaxIterations(3);
  tracker_->SetTranslationEnabled(false);
  tracker_->SetKeyframe(quarter_keyframe_);
  tracker_->Track(quarter_frame);

  tracker_->SetMaxIterations(5);
  tracker_->SetTranslationEnabled(true);
  half_frame.Tcw = quarter_frame.Tcw;
  tracker_->SetKeyframe(half_keyframe_);
  tracker_->Track(half_frame);

  tracker_->SetMaxIterations(7);
  tracker_->SetTranslationEnabled(true);
  frame.Tcw = half_frame.Tcw;
  tracker_->SetKeyframe(keyframe_);
  tracker_->Track(frame);
}

template class PyramidTracker<ColorTracker>;
template class PyramidTracker<DepthTracker>;
template class PyramidTracker<LightTracker>;

} // namespace vulcan