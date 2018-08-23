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
      // Vector3f result;
      // const float scale = 1.0f / 32767.0f;
      // result[0] = scale * color[0];
      // result[1] = scale * color[1];
      // result[2] = scale * color[2];
      // return result;

      return color;
    }

    VULCAN_HOST_DEVICE
    inline void SetColor(const Vector3f& c)
    {
      // color[0] = 32767 * c[0];
      // color[1] = 32767 * c[1];
      // color[2] = 32767 * c[2];

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