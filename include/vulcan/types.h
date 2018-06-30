#pragma once

namespace vulcan
{

enum Visibility : uint8_t
{
  VISIBILITY_UNKNOWN = 0,
  VISIBILITY_FALSE   = 1,
  VISIBILITY_TRUE    = 2,
};

enum AllocationType : uint8_t
{
  ALLOC_TYPE_NONE   = 0,
  ALLOC_TYPE_MAIN   = 1,
  ALLOC_TYPE_EXCESS = 2,
};

} // namespace vulcan