#pragma once

#include <vulcan/integrator.h>

namespace vulcan
{

class DepthIntegrator : public Integrator
{
  public:

    DepthIntegrator(std::shared_ptr<Volume> volume);

    void Integrate(const Frame& frame) override;
};

} // namespace vulcan