#pragma once

#include <memory>

namespace vulcan
{

class Frame;
class Volume;

class Tracker
{
  public:

    Tracker(std::shared_ptr<Volume> volume);

    std::shared_ptr<Volume> GetVolume() const;

    void Track(Frame& frame);

  protected:

    std::shared_ptr<Volume> volume_;
};

} // namespace vulcan