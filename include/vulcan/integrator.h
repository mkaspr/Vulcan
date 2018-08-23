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

    float GetMaxWeight() const;

    void SetMaxWeight(float weight);

    virtual void Integrate(const Frame& frame) = 0;

  protected:

    std::shared_ptr<Volume> volume_;

    float max_weight_;
};

} // namespace vulcan