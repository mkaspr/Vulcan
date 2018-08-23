#pragma once

#include <memory>

namespace vulcan
{

class Frame;
class Volume;

class Integrator
{
  public:

    Integrator(std::shared_ptr<Volume> volume);

    std::shared_ptr<Volume> GetVolume() const;

    float GetMaxDistanceWeight() const;

    void SetMaxDistanceWeight(float weight);

    float GetMaxColorWeight() const;

    void SetMaxColorWeight(float weight);

    virtual void Integrate(const Frame& frame) = 0;

  protected:

    std::shared_ptr<Volume> volume_;

    float max_distance_weight_;

    float max_color_weight_;
};

} // namespace vulcan