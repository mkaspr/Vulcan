#pragma once

#include <memory>
#include <vulcan/matrix.h>

namespace vulcan
{

class Frame;
class Volume;

class Integrator
{
  public:

    Integrator(std::shared_ptr<Volume> volume);

    std::shared_ptr<Volume> GetVolume() const;

    const Vector2f& GetDepthRange() const;

    void SetDepthRange(const Vector2f& range);

    void SetDepthRange(float min, float max);

    float GetMaxDistanceWeight() const;

    void SetMaxDistanceWeight(float weight);

    float GetMaxColorWeight() const;

    void SetMaxColorWeight(float weight);

    virtual void Integrate(const Frame& frame) = 0;

  protected:

    std::shared_ptr<Volume> volume_;

    Vector2f depth_range_;

    float max_distance_weight_;

    float max_color_weight_;
};

} // namespace vulcan