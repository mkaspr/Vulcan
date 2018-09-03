#pragma once

#include <vulcan/integrator.h>
#include <vulcan/image.h>
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

    void ComputeFrameMask(const Frame& frame);

    void IntegrateDepth(const Frame& frame);

    void IntegrateColor(const Frame& frame);

  protected:

    Light light_;

    Image frame_mask_;

    float depth_threshold_;
};

} // namespace vulcan