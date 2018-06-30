#pragma once

#include <vulcan/device.h>
#include <vulcan/matrix.h>

namespace vulcan
{

class Block
{
  public:

    static const int resolution = 8;

    static const int voxel_count = 512;

  public:

    VULCAN_HOST_DEVICE
    Block() :
      origin_(0, 0, 0)
    {
    }

    VULCAN_HOST_DEVICE
    Block(short x, short y, short z) :
      origin_(x, y, z)
    {
    }

    VULCAN_HOST_DEVICE
    Block(const Vector3s& origin) :
      origin_(origin)
    {
    }

    VULCAN_HOST_DEVICE
    inline const Vector3s& GetOrigin() const
    {
      return origin_;
    }

    VULCAN_HOST_DEVICE
    inline bool operator==(const Block& block) const
    {
      return origin_ == block.origin_;
    }

    VULCAN_HOST_DEVICE
    inline bool operator!=(const Block& block) const
    {
      return !(origin_ == block.origin_);
    }

    // index -> global voxel position

    // x y z -> index

    // index -> x y z

    // static: float3 -> block

    // get voxel count

  protected:

    Vector3s origin_;

  private:

    short pad_;
};

} // namespace vulcan