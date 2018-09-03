#pragma once

#include <vulcan/integrator.h>

namespace vulcan
{

class ColorIntegrator : public Integrator
{
  public:

    ColorIntegrator(std::shared_ptr<Volume> volume);

    void Integrate(const Frame& frame) override;

  protected:

    void IntegrateDepth(const Frame& frame);

    void IntegrateColor(const Frame& frame);
};

} // namespace vulcan