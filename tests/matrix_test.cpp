#include <gtest/gtest.h>
#include <vulcan/matrix.h>

namespace vulcan
{
namespace testing
{

TEST(Vector, Constructor)
{
  Vector4f a(0, 1, 2, 3);
  ASSERT_FLOAT_EQ(0, a[0]);
  ASSERT_FLOAT_EQ(1, a[1]);
  ASSERT_FLOAT_EQ(2, a[2]);
  ASSERT_FLOAT_EQ(3, a[3]);

  Vector3f b(a);
  ASSERT_FLOAT_EQ(0, b[0]);
  ASSERT_FLOAT_EQ(1, b[1]);
  ASSERT_FLOAT_EQ(2, b[2]);
}

TEST(Vector, SquaredNorm)
{
  Vector3f vec3;
  float expected;

  vec3[0] =  1.5;
  vec3[1] = -0.5;
  vec3[2] =  3.7;

  expected = 0;
  expected += vec3[0] * vec3[0];
  expected += vec3[1] * vec3[1];
  expected += vec3[2] * vec3[2];
  ASSERT_FLOAT_EQ(expected, vec3.SquaredNorm());
}

TEST(Vector, Norm)
{
  Vector3f vec3;
  float expected;

  vec3[0] =  1.5;
  vec3[1] = -0.5;
  vec3[2] =  3.7;

  expected = 0;
  expected += vec3[0] * vec3[0];
  expected += vec3[1] * vec3[1];
  expected += vec3[2] * vec3[2];
  expected = sqrtf(expected);
  ASSERT_FLOAT_EQ(expected, vec3.Norm());
}

TEST(Vector, Normalize)
{
  float norm;
  Vector3f found;
  Vector3f expected;

  found[0] =  1.5;
  found[1] = -0.5;
  found[2] =  3.7;

  norm = 0;
  norm += found[0] * found[0];
  norm += found[1] * found[1];
  norm += found[2] * found[2];
  norm = sqrtf(norm);

  expected[0] = found[0] / norm;
  expected[1] = found[1] / norm;
  expected[2] = found[2] / norm;

  found.Normalize();

  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);
  ASSERT_FLOAT_EQ(expected[2], found[2]);
  ASSERT_FLOAT_EQ(1, found.Norm());
}

TEST(Vector, Normalized)
{
  float norm;
  Vector3f vec3;
  Vector3f found;
  Vector3f expected;

  vec3[0] =  1.5;
  vec3[1] = -0.5;
  vec3[2] =  3.7;

  norm = 0;
  norm += vec3[0] * vec3[0];
  norm += vec3[1] * vec3[1];
  norm += vec3[2] * vec3[2];
  norm = sqrtf(norm);

  expected[0] = vec3[0] / norm;
  expected[1] = vec3[1] / norm;
  expected[2] = vec3[2] / norm;

  found = vec3.Normalized();

  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);
  ASSERT_FLOAT_EQ(expected[2], found[2]);
  ASSERT_FLOAT_EQ(1, found.Norm());
}

} // namespace testing

} // namespace vulcan