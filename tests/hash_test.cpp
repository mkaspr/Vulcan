#include <gtest/gtest.h>
#include <vulcan/hash.h>

namespace vulcan
{
namespace testing
{

TEST(HashEntry, Size)
{
  ASSERT_EQ(16, sizeof(HashEntry));
}

TEST(HashEntry, Constructor)
{
  HashEntry entry;
  ASSERT_FALSE(entry.IsAllocated());
  ASSERT_FALSE(entry.HasNext());
}

TEST(HashEntry, IsAllocated)
{
  HashEntry entry;

  entry.data = 0;
  ASSERT_TRUE(entry.IsAllocated());

  entry.data = HashEntry::invalid;
  ASSERT_FALSE(entry.IsAllocated());

  entry.data = 32;
  ASSERT_TRUE(entry.IsAllocated());

  entry.InvalidateData();
  ASSERT_FALSE(entry.IsAllocated());
}

TEST(HashEntry, HasNext)
{
  HashEntry entry;

  entry.next = 0;
  ASSERT_TRUE(entry.HasNext());

  entry.next = HashEntry::invalid;
  ASSERT_FALSE(entry.HasNext());

  entry.next = 32;
  ASSERT_TRUE(entry.HasNext());

  entry.InvalidateNext();
  ASSERT_FALSE(entry.HasNext());
}

} // namespace testing

} // namespace vulcan