#include <vulcan/integrator.h>
#include <vulcan/exception.h>

namespace vulcan
{

Integrator::Integrator(std::shared_ptr<Volume> volume) :
  volume_(volume),
  max_weight_(16)
{
}

std::shared_ptr<Volume> Integrator::GetVolume() const
{
  return volume_;
}

float Integrator::GetMaxWeight() const
{
  return max_weight_;
}

void Integrator::SetMaxWeight(float weight)
{
  VULCAN_DEBUG(weight > 0);
  max_weight_ = weight;
}

} // namespace vulcan