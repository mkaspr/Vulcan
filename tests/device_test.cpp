#include <gtest/gtest.h>
#include <vulcan/device.h>

namespace vulcan
{
namespace testing
{

TEST(Device, GetKernelBlocks)
{
  dim3 total, threads, blocks;

  ASSERT_EQ(1, GetKernelBlocks(1, 1));
  ASSERT_EQ(2, GetKernelBlocks(2, 1));
  ASSERT_EQ(1, GetKernelBlocks(32, 32));
  ASSERT_EQ(2, GetKernelBlocks(33, 32));
  ASSERT_EQ(2, GetKernelBlocks(63, 32));
  ASSERT_EQ(2, GetKernelBlocks(64, 32));
  ASSERT_EQ(3, GetKernelBlocks(65, 32));

  total = dim3(1, 1, 1);
  threads = dim3(1, 1, 1);
  blocks = GetKernelBlocks(total, threads);
  ASSERT_EQ(1, blocks.x);
  ASSERT_EQ(1, blocks.y);
  ASSERT_EQ(1, blocks.z);

  total = dim3(32, 127, 7);
  threads = dim3(32, 64, 2);
  blocks = GetKernelBlocks(total, threads);
  ASSERT_EQ(1, blocks.x);
  ASSERT_EQ(2, blocks.y);
  ASSERT_EQ(4, blocks.z);
}

} // namespace testing

} // namespace vulcan