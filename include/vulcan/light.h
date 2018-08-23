#pragma once

#include <vulcan/exception.h>
#include <vulcan/device.h>
#include <vulcan/matrix.h>

namespace vulcan
{

class Light
{
  public:

    VULCAN_HOST_DEVICE
    Light() :
      intensity_(1.0f),
      position_(0, 0, 0)
    {
    }

    VULCAN_HOST_DEVICE
    inline float GetIntensity() const
    {
      return intensity_;
    }

    VULCAN_HOST_DEVICE
    inline void SetIntensity(float intensity)
    {
      VULCAN_ASSERT(intensity >= 0);
      intensity_ = intensity;
    }

    VULCAN_HOST_DEVICE
    inline const Vector3f& GetPosition() const
    {
      return position_;
    }

    VULCAN_HOST_DEVICE
    inline void SetPosition(const Vector3f& position)
    {
      position_ = position;
    }

    VULCAN_HOST_DEVICE
    inline void SetPosition(float x, float y, float z)
    {
      SetPosition(Vector3f(x, y, z));
    }

    VULCAN_HOST_DEVICE
    inline float GetShading(const Vector3f& point, const Vector3f& normal) const
    {
      const Vector3f delta = position_ - point;
      const Vector3f direction = delta.Normalized();
      const float distance_squared = delta.SquaredNorm();
      return intensity_ * normal.Dot(direction) / distance_squared;
    }

  protected:

    float intensity_;

    Vector3f position_;
};

} // namespace vulcan