#include <gtest/gtest.h>
#include <vulcan/voxel.h>

namespace vulcan
{
namespace testing
{

TEST(Voxel, Empty)
{
  Voxel voxel = Voxel::Empty();

  ASSERT_EQ(1, voxel.distance);
  ASSERT_EQ(0, voxel.color[0]);
  ASSERT_EQ(0, voxel.color[1]);
  ASSERT_EQ(0, voxel.color[2]);
  ASSERT_EQ(0, voxel.distance_weight);
  ASSERT_EQ(0, voxel.color_weight);
}

TEST(Voxel, Color)
{
  Voxel voxel;
  Vector3f found;
  Vector3f expected;

  expected[0] = 0.8;
  expected[1] = 0.1;
  expected[2] = 0.4;

  voxel.SetColor(expected);
  found = voxel.GetColor();

  const float epsilon = 1.0f / std::pow(2, 15);
  ASSERT_NEAR(expected[0], found[0], epsilon);
  ASSERT_NEAR(expected[1], found[1], epsilon);
  ASSERT_NEAR(expected[2], found[2], epsilon);
}

} // namespace testing

} // namespace vulcan