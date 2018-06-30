#pragma once

#include <climits>
#include <vulcan/block.h>
#include <vulcan/device.h>

namespace vulcan
{

class HashEntry
{
  public:

    static const int invalid = -1;

  public:

    VULCAN_HOST_DEVICE
    HashEntry() :
      data(invalid),
      next(invalid)
    {
    }

    VULCAN_HOST_DEVICE
    inline bool IsAllocated() const
    {
      return data != invalid;
    }

    VULCAN_HOST_DEVICE
    inline void InvalidateData()
    {
      data = invalid;
    }

    VULCAN_HOST_DEVICE
    inline bool HasNext() const
    {
      return next != invalid;
    }

    VULCAN_HOST_DEVICE
    inline void InvalidateNext()
    {
      next = invalid;
    }

  public:

    Block block;

    int data;

    int next;
};


} // namespace vulcan