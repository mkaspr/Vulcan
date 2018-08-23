#pragma once

#include <vulcan/device.h>
#include <vulcan/matrix.h>

namespace vulcan
{

class Voxel
{
  public:

    VULCAN_HOST_DEVICE
    Voxel()
    {
    }

    VULCAN_HOST_DEVICE
    inline Vector3f GetColor() const
    {
      Vector3f result;
      const float scale = 1.0f / 32768.0f;
      result[0] = scale * color[0];
      result[1] = scale * color[1];
      result[2] = scale * color[2];
      return result;
    }

    VULCAN_HOST_DEVICE
    inline void SetColor(const Vector3f& c)
    {
      color[0] = 32768 * c[0];
      color[1] = 32768 * c[1];
      color[2] = 32768 * c[2];
    }

    VULCAN_HOST_DEVICE
    static inline Voxel Empty()
    {
      Voxel result;
      result.weight = 0;
      result.distance = 1;
      result.color = Vector4s::Zeros();
      return result;
    }

  public:

    float weight;

    float distance;

    Vector4s color;
};

} // namespace vulcan