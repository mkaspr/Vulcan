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

  keyframe_->Downsample(*half_keyframe_);
  half_keyframe_->Downsample(*quarter_keyframe_);

  tracker_->SetMaxIterations(5);
  tracker_->SetTranslationEnabled(false);
  tracker_->SetKeyframe(quarter_keyframe_);
  tracker_->Track(quarter_frame);

  tracker_->SetMaxIterations(5);
  tracker_->SetTranslationEnabled(true);
  tracker_->SetKeyframe(quarter_keyframe_);
  tracker_->Track(quarter_frame);

  tracker_->SetMaxIterations(10);
  tracker_->SetTranslationEnabled(true);
  half_frame.Twc = quarter_frame.Twc;
  tracker_->SetKeyframe(half_keyframe_);
  tracker_->Track(half_frame);

  tracker_->SetMaxIterations(15);
  tracker_->SetTranslationEnabled(true);
  frame.Twc = half_frame.Twc;
  tracker_->SetKeyframe(keyframe_);
  tracker_->Track(frame);

  // Matrix3f R;

  // R(0, 0) = 1.00000070;
  // R(1, 0) = 0.00021569;
  // R(2, 0) = -0.00036980;

  // R(0, 1) = -0.00021233;
  // R(1, 1) = 0.99994549;
  // R(2, 1) = 0.01043146;

  // R(0, 2) = 0.00037226;
  // R(1, 2) = -0.01043150;
  // R(2, 2) = 0.99994517;

  // Vector3f t;

  // t[0] = 0.00045346;
  // t[1] = 0.00161962;
  // t[2] = 0.01716002;

  // const Transform Tinc = Transform::Translate(t) * Transform::Rotate(R);
  // frame.Twc = keyframe_->Twc;// * Tinc;

  // tracker_->SetMaxIterations(50);
  // tracker_->SetTranslationEnabled(true);
  // tracker_->SetKeyframe(keyframe_);
  // tracker_->Track(frame);
}

template class PyramidTracker<ColorTracker>;
template class PyramidTracker<DepthTracker>;
template class PyramidTracker<LightTracker>;

} // namespace vulcan