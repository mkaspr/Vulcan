#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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

TYPED_TEST(Buffer, CopyFromDevice)
{
  vulcan::Buffer<TypeParam> buffer;
  thrust::host_vector<TypeParam> expected(64);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    expected[i] = TypeParam(i);
  }

  thrust::device_vector<TypeParam> d_expected(expected);
  buffer.Resize(expected.size());
  buffer.CopyFromDevice(d_expected.data().get());

  thrust::device_ptr<const TypeParam> pointer(buffer.GetData());
  thrust::host_vector<TypeParam> found(pointer, pointer + expected.size());

  for (size_t i = 0; i < expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TYPED_TEST(Buffer, CopyToDevice)
{
  vulcan::Buffer<TypeParam> buffer;
  thrust::host_vector<TypeParam> expected(64);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    expected[i] = TypeParam(i);
  }

  buffer.Resize(expected.size());
  thrust::device_ptr<TypeParam> pointer(buffer.GetData());
  thrust::copy(expected.begin(), expected.end(), pointer);
  thrust::device_vector<TypeParam> d_found(expected);
  buffer.CopyToDevice(d_found.data().get());
  thrust::host_vector<TypeParam> found(d_found);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TYPED_TEST(Buffer, CopyFromHost)
{
  vulcan::Buffer<TypeParam> buffer;
  thrust::host_vector<TypeParam> expected(64);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    expected[i] = TypeParam(i);
  }

  buffer.Resize(expected.size());
  buffer.CopyFromHost(expected.data());

  thrust::device_ptr<const TypeParam> pointer(buffer.GetData());
  thrust::host_vector<TypeParam> found(pointer, pointer + expected.size());

  for (size_t i = 0; i < expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TYPED_TEST(Buffer, CopyToHost)
{
  vulcan::Buffer<TypeParam> buffer;
  thrust::host_vector<TypeParam> expected(64);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    expected[i] = TypeParam(i);
  }

  buffer.Resize(expected.size());
  thrust::device_ptr<TypeParam> pointer(buffer.GetData());
  thrust::copy(expected.begin(), expected.end(), pointer);
  std::vector<TypeParam> found(expected.size());
  buffer.CopyToHost(found.data());

  for (size_t i = 0; i < expected.size(); ++i)
  {
    ASSERT_EQ(expected[i], found[i]);
  }
}

TYPED_TEST(Buffer, IsEmpty)
{
  vulcan::Buffer<TypeParam> buffer;
  ASSERT_TRUE(buffer.IsEmpty());

  buffer.Reserve(16);
  ASSERT_TRUE(buffer.IsEmpty());

  buffer.Resize(16);
  ASSERT_FALSE(buffer.IsEmpty());

  buffer.Reserve(32);
  ASSERT_FALSE(buffer.IsEmpty());

  buffer.Resize(0);
  ASSERT_TRUE(buffer.IsEmpty());

  buffer.Resize(64);
  ASSERT_FALSE(buffer.IsEmpty());
}

} // namespace testing

} // namespace vulcan