#pragma once

#include <cmath>
#include <vulcan/device.h>

namespace vulcan
{

template <typename T>
VULCAN_HOST_DEVICE
inline T min(T a, T b)
{
  return (b < a) ? b : a;
}

template <typename T>
VULCAN_HOST_DEVICE
inline T sqrt(T value)
{
  return ::std::sqrt(value);
}

#ifdef __CUDA_ARCH__

template <typename T>
VULCAN_HOST_DEVICE
inline bool isnan(T value)
{
  return ::isnan(value);
}

#else

template <typename T>
VULCAN_HOST_DEVICE
inline bool isnan(T value)
{
  return ::std::isnan(value);
}

#endif

} // namespace vulcan