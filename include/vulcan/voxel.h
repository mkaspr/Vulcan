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
    static inline Voxel Empty()
    {
      Voxel result;
      result.weight = 0;
      result.distance = 1;
      result.color = Vector3f::Zeros();
      return result;
    }

  public:

    float weight;

    float distance;

    Vector3f color;
};

} // namespace vulcan