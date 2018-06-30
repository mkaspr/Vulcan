#pragma once

#include <cuda_runtime.h>
#include <vulcan/exception.h>

#define VULCAN_GLOBAL      __global__
#define VULCAN_SHARED      __shared__
#define VULCAN_HOST        __host__
#define VULCAN_DEVICE      __device__
#define VULCAN_HOST_DEVICE __host__ __device__

#define CUDA_ASSERT(cmd) {                                                     \
  const cudaError_t code = cmd;                                                \
  VULCAN_ASSERT_MSG(code == cudaSuccess, ::vulcan::GetCudaErrorString(code));  \
}

#define CUDA_ASSERT_LAST() {                                                   \
  CUDA_DEBUG(cudaDeviceSynchronize());                                         \
  CUDA_DEBUG(cudaGetLastError());                                              \
}

#define CUDA_DEBUG_LAST() {                                                    \
  CUDA_DEBUG(cudaDeviceSynchronize());                                         \
  CUDA_DEBUG(cudaGetLastError());                                              \
}

#ifdef NDEBUG

#define CUDA_DEBUG(cmd) cmd;

#define CUDA_LAUNCH(kernel, grids, blocks, shared, stream, ...)                \
  kernel<<<grids, blocks, shared, stream>>>(__VA_ARGS__)

#else

#define CUDA_DEBUG CUDA_ASSERT

#define CUDA_LAUNCH(kernel, blocks, threads, shared, stream, ...) {            \
  kernel<<<blocks, threads, shared, stream>>>(__VA_ARGS__);                    \
  CUDA_DEBUG(cudaDeviceSynchronize());                                         \
  CUDA_DEBUG(cudaGetLastError());                                              \
}

#endif

#ifdef __CUDA_ARCH__
#define VULCAN_DEVICE_RETURN(value) return value
#else
#define VULCAN_DEVICE_RETURN(value)
#endif

namespace vulcan
{

#ifdef __CUDA_ARCH__

inline const char* GetCudaErrorString(cudaError_t code)
{
  return cudaGetErrorString(code);
}

#else

inline std::string GetCudaErrorString(cudaError_t code)
{
  const std::string error = cudaGetErrorString(code);
  return error + " [cuda error " + std::to_string(code) + "]";
}

#endif

inline size_t GetKernelBlocks(size_t total, size_t threads)
{
  return (total + threads - 1) / threads;
}

inline dim3 GetKernelBlocks(const dim3& total, const dim3& threads)
{
  dim3 blocks;
  blocks.x = GetKernelBlocks(total.x, threads.x);
  blocks.y = GetKernelBlocks(total.y, threads.y);
  blocks.z = GetKernelBlocks(total.z, threads.z);
  return blocks;
}

} //  namespace vulcan