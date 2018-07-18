#pragma once

#include <vulcan/device.h>

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
      result.distance = -1;
      result.weight = 0;
      return result;
    }

  public:

    float distance;

    float weight;
};

} // namespace vulcan