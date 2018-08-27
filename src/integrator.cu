#include <vulcan/integrator.h>
#include <vulcan/exception.h>

namespace vulcan
{

Integrator::Integrator(std::shared_ptr<Volume> volume) :
  volume_(volume),
  depth_range_(0.1f, 5.0f),
  max_distance_weight_(16),
  max_color_weight_(16)
{
}

std::shared_ptr<Volume> Integrator::GetVolume() const
{
  return volume_;
}

const Vector2f& Integrator::GetDepthRange() const
{
  return depth_range_;
}

void Integrator::SetDepthRange(const Vector2f& range)
{
  VULCAN_DEBUG(range[0] > 0 && range[0] < range[1]);
  depth_range_ = range;
}

void Integrator::SetDepthRange(float min, float max)
{
  SetDepthRange(Vector2f(min, max));
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