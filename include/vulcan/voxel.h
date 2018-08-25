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
    inline const Vector3f& GetColor() const
    {
      return color;
    }

    VULCAN_HOST_DEVICE
    inline void SetColor(const Vector3f& c)
    {
      color = c;
    }

    VULCAN_HOST_DEVICE
    static inline Voxel Empty()
    {
      Voxel result;
      result.distance = 1;
      result.color = Vector3f::Zeros();
      result.distance_weight = 0;
      result.color_weight = 0;
      return result;
    }

  public:

    float distance;

    Vector3f color;

    short distance_weight;

    short color_weight;
};

} // namespace vulcan