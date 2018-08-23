#include <vulcan/integrator.h>
#include <vulcan/exception.h>

namespace vulcan
{

Integrator::Integrator(std::shared_ptr<Volume> volume) :
  volume_(volume),
  max_distance_weight_(16),
  max_color_weight_(16)
{
}

std::shared_ptr<Volume> Integrator::GetVolume() const
{
  return volume_;
}

float Integrator::GetMaxDistanceWeight() const
{
  return max_distance_weight_;
}

void Integrator::SetMaxDistanceWeight(float weight)
{
  VULCAN_DEBUG(weight > 0);
  max_distance_weight_ = weight;
}

float Integrator::GetMaxColorWeight() const
{
  return max_color_weight_;
}

void Integrator::SetMaxColorWeight(float weight)
{
  VULCAN_DEBUG(weight > 0);
  max_color_weight_ = weight;
}

} // namespace vulcan