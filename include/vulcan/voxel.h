#pragma once

#include <cstdint>

namespace vulcan
{

class Voxel
{
  public:

    Voxel() :
      distance(0),
      weight(0)
    {
    }

  public:

    float distance;

    float weight;
};

} // namespace vulcan