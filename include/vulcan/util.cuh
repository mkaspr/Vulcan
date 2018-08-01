#pragma once

#include <vulcan/device.h>

namespace vulcan
{

template <int BLOCK_SIZE>
VULCAN_DEVICE
inline int PrefixSum(int value, int thread, int& total)
{
  VULCAN_SHARED int block_offset;
  VULCAN_SHARED int buffer[BLOCK_SIZE];
  buffer[thread] = value;

  __syncthreads();

  int i, j;

  for (i = 1, j = 1; i < BLOCK_SIZE; i <<= 1)
  {
    j |= i;

    if ((thread & j) == j)
    {
      buffer[thread] += buffer[thread - i];
    }

    __syncthreads();
  }

  for (i >>= 2, j >>= 1; i >= 1; i >>= 1, j >>= 1)
  {
    if (thread != BLOCK_SIZE - 1 && (thread & j) == j)
    {
      buffer[thread + i] += buffer[thread];
    }

    __syncthreads();
  }

  if (thread == 0 && buffer[BLOCK_SIZE - 1] > 0)
  {
    block_offset = atomicAdd(&total, buffer[BLOCK_SIZE - 1]);
  }

  __syncthreads();

  const int prev_end = (thread == 0) ? 0 : buffer[thread - 1];
  return (buffer[thread] == prev_end) ? -1 : block_offset + prev_end;
}

template <int MAX_BLOCK_SIZE>
VULCAN_DEVICE
inline int PrefixSum(int value, int thread, int& total, int block_size)
{
  VULCAN_SHARED int block_offset;
  VULCAN_SHARED int buffer[MAX_BLOCK_SIZE];
  buffer[thread] = value;

  __syncthreads();

  int i, j;

  for (i = 1, j = 1; i < block_size; i <<= 1)
  {
    j |= i;

    if ((thread & j) == j)
    {
      buffer[thread] += buffer[thread - i];
    }

    __syncthreads();
  }

  for (i >>= 2, j >>= 1; i >= 1; i >>= 1, j >>= 1)
  {
    if (thread != block_size - 1 && (thread & j) == j)
    {
      buffer[thread + i] += buffer[thread];
    }

    __syncthreads();
  }

  if (thread == 0 && buffer[block_size - 1] > 0)
  {
    block_offset = atomicAdd(&total, buffer[block_size - 1]);
  }

  __syncthreads();

  const int prev_end = (thread == 0) ? 0 : buffer[thread - 1];
  return (buffer[thread] == prev_end) ? -1 : block_offset + prev_end;
}

} // namespace vulcan
