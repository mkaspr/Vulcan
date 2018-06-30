#include <gtest/gtest.h>
#include <vulcan/buffer.h>

namespace vulcan
{
namespace testing
{

template <typename T>
class Buffer : public ::testing::Test {};
typedef ::testing::Types<uint8_t, uint16_t, int, float> Types;
TYPED_TEST_CASE(Buffer, Types);

TYPED_TEST(Buffer, Constructor)
{
  vulcan::Buffer<TypeParam> a;
  ASSERT_EQ(nullptr, a.GetData());
  ASSERT_EQ(0, a.GetCapacity());
  ASSERT_EQ(0, a.GetSize());

  vulcan::Buffer<TypeParam> b(10);
  ASSERT_NE(nullptr, b.GetData());
  ASSERT_EQ(10, b.GetCapacity());
  ASSERT_EQ(10, b.GetSize());
}

TYPED_TEST(Buffer, Resize)
{
  vulcan::Buffer<TypeParam> buffer;
  TypeParam* pointer;

  buffer.Resize(8);
  pointer = buffer.GetData();
  ASSERT_NE(nullptr, buffer.GetData());
  ASSERT_EQ(8, buffer.GetCapacity());
  ASSERT_EQ(8, buffer.GetSize());

  buffer.Resize(4);
  ASSERT_EQ(pointer, buffer.GetData());
  ASSERT_EQ(8, buffer.GetCapacity());
  ASSERT_EQ(4, buffer.GetSize());

  buffer.Resize(16);
  ASSERT_NE(nullptr, buffer.GetData());
  ASSERT_EQ(16, buffer.GetCapacity());
  ASSERT_EQ(16, buffer.GetSize());
}

TYPED_TEST(Buffer, Reserve)
{
  vulcan::Buffer<TypeParam> buffer;
  TypeParam* pointer;

  buffer.Reserve(8);
  pointer = buffer.GetData();
  ASSERT_NE(nullptr, buffer.GetData());
  ASSERT_EQ(8, buffer.GetCapacity());
  ASSERT_EQ(0, buffer.GetSize());

  buffer.Reserve(4);
  ASSERT_EQ(pointer, buffer.GetData());
  ASSERT_EQ(8, buffer.GetCapacity());
  ASSERT_EQ(0, buffer.GetSize());

  buffer.Reserve(16);
  ASSERT_NE(nullptr, buffer.GetData());
  ASSERT_EQ(16, buffer.GetCapacity());
  ASSERT_EQ(0, buffer.GetSize());
}

} // namespace testing

} // namespace vulcan