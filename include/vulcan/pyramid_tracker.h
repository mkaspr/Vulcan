#pragma once

#include <memory>

namespace vulcan
{

class Frame;

template <typename Tracker>
class PyramidTracker
{
  public:

    PyramidTracker();

    virtual ~PyramidTracker();

    std::shared_ptr<const Frame> GetKeyframe() const;

    void SetKeyframe(std::shared_ptr<const Frame> keyframe);

    void Track(Frame& frame);

  protected:

    std::unique_ptr<Tracker> tracker_;

    std::shared_ptr<const Frame> keyframe_;

    std::shared_ptr<Frame> half_keyframe_;

    std::shared_ptr<Frame> quarter_keyframe_;
};

} // namespace vulcan