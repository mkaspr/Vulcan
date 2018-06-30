#include <gtest/gtest.h>
#include <vulcan/block.h>

namespace vulcan
{
namespace testing
{

TEST(Block, Size)
{
  ASSERT_EQ(8, sizeof(Block));
}


} // namespace testing

} // namespace vulcan