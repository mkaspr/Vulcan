#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vulcan/util.cuh>

namespace vulcan
{
namespace testing
{

class Util : public ::testing::TestWithParam<int> {};

VULCAN_DEVICE int prefix_total;

template <int BLOCK_SIZE>
VULCAN_GLOBAL
void PrefixSumTestKernel(int* output, int count)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  int value = 0;

  if (index < count)
  {
    value = index % 7;
  }

  const int offset = PrefixSum<BLOCK_SIZE>(value, threadIdx.x, prefix_total);

  for (int i = 0; i < value; ++i)
  {
    output[offset + i] = index;
  }
}

inline void ResetPrefixTotal()
{
  const int result = 0;
  const size_t bytes = sizeof(int);
  CUDA_ASSERT(cudaMemcpyToSymbol(prefix_total, &result, bytes));
}

inline int GetPrefixTotal()
{
  int result;
  const size_t bytes = sizeof(int);
  CUDA_ASSERT(cudaMemcpyFromSymbol(&result, prefix_total, bytes));
  return result;
}

inline int GetExpectedCounts(std::vector<int>& expected)
{
  int total = 0;

  for (size_t i = 0; i < expected.size(); ++i)
  {
    expected[i] = i % 7;
    total += expected[i];
  }

  return total;
}

TEST_P(Util, PrefixSum)
{
  ResetPrefixTotal();

  const size_t count = GetParam();
  std::vector<int> expected(count);
  const int expected_total = GetExpectedCounts(expected);

  const size_t threads = 512;
  const size_t blocks = GetKernelBlocks(count, threads);
  thrust::device_vector<int> d_found(expected_total);
  int* ptr = thrust::device_pointer_cast(d_found.data()).get();
  CUDA_LAUNCH(PrefixSumTestKernel<512>, blocks, threads, 0, 0, ptr, count);

  ASSERT_EQ(expected_total, GetPrefixTotal());

  thrust::host_vector<int> h_found(expected_total);
  thrust::copy(d_found.begin(), d_found.end(), h_found.begin());
  int prev_index;

  for (size_t i = 0; i < h_found.size(); ++i)
  {
    const int index = h_found[i];

    if (i > 0 && prev_index != index)
    {
      ASSERT_EQ(0, expected[prev_index]);
    }

    --expected[index];
    prev_index = index;
  }
}

INSTANTIATE_TEST_CASE_P( , Util,
    ::testing::Values(1, 2, 31, 32, 33, 511, 512, 513, 1023, 1024, 1025));

} // namespace testing

} // namespace vulcan