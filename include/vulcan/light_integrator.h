#pragma once

#include <vulcan/integrator.h>
#include <vulcan/light.h>

namespace vulcan
{

class LightIntegrator : public Integrator
{
  public:

    LightIntegrator(std::shared_ptr<Volume> volume);

    const Light& GetLight() const;

    void SetLight(const Light& light);

    void Integrate(const Frame& frame) override;

  protected:

    Light light_;
};

} // namespace vulcan