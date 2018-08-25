#include <gtest/gtest.h>
#include <vulcan/matrix.h>

namespace vulcan
{
namespace testing
{

TEST(Vector, Constructor)
{
  const Vector2f a(0, 1);
  ASSERT_EQ(2, a.GetRows());
  ASSERT_EQ(1, a.GetColumns());
  ASSERT_EQ(2, a.GetTotal());
  ASSERT_FLOAT_EQ(0, a[0]);
  ASSERT_FLOAT_EQ(1, a[1]);

  const Vector3f b(0, 1, 2);
  ASSERT_EQ(3, b.GetRows());
  ASSERT_EQ(1, b.GetColumns());
  ASSERT_FLOAT_EQ(0, b[0]);
  ASSERT_FLOAT_EQ(1, b[1]);
  ASSERT_FLOAT_EQ(2, b[2]);

  const Vector4f c(0, 1, 2, 3);
  ASSERT_EQ(4, c.GetRows());
  ASSERT_EQ(1, c.GetColumns());
  ASSERT_EQ(4, c.GetTotal());
  ASSERT_FLOAT_EQ(0, c[0]);
  ASSERT_FLOAT_EQ(1, c[1]);
  ASSERT_FLOAT_EQ(2, c[2]);
  ASSERT_FLOAT_EQ(3, c[3]);

  const Vector3f d(c);
  ASSERT_EQ(3, d.GetRows());
  ASSERT_EQ(1, d.GetColumns());
  ASSERT_EQ(3, d.GetTotal());
  ASSERT_FLOAT_EQ(c[0], d[0]);
  ASSERT_FLOAT_EQ(c[1], d[1]);
  ASSERT_FLOAT_EQ(c[2], d[2]);

  const Vector3i e(d);
  ASSERT_EQ(3, e.GetRows());
  ASSERT_EQ(1, e.GetColumns());
  ASSERT_EQ(3, e.GetTotal());
  ASSERT_NEAR(d[0], e[0], 1E-8);
  ASSERT_NEAR(d[1], e[1], 1E-8);
  ASSERT_NEAR(d[2], e[2], 1E-8);

  const Vector4f f(d);
  ASSERT_EQ(4, f.GetRows());
  ASSERT_EQ(1, f.GetColumns());
  ASSERT_EQ(4, f.GetTotal());
  ASSERT_FLOAT_EQ(d[0], f[0]);
  ASSERT_FLOAT_EQ(d[1], f[1]);
  ASSERT_FLOAT_EQ(d[2], f[2]);
  ASSERT_FLOAT_EQ(   0, f[3]);
}

TEST(Vector, SquaredNorm)
{
  float expected;

  ASSERT_FLOAT_EQ(2, Vector2f::Ones().SquaredNorm());
  ASSERT_FLOAT_EQ(3, Vector3f::Ones().SquaredNorm());
  ASSERT_FLOAT_EQ(4, Vector4f::Ones().SquaredNorm());

  ASSERT_FLOAT_EQ(0, Vector2f::Zeros().SquaredNorm());
  ASSERT_FLOAT_EQ(0, Vector3f::Zeros().SquaredNorm());
  ASSERT_FLOAT_EQ(0, Vector4f::Zeros().SquaredNorm());

  Vector3f a;
  a[0] =  1.5;
  a[1] = -0.5;
  a[2] =  3.7;

  expected = 0;
  expected += a[0] * a[0];
  expected += a[1] * a[1];
  expected += a[2] * a[2];
  ASSERT_FLOAT_EQ(expected, a.SquaredNorm());

  Vector4f b;
  b[0] =  1.5;
  b[1] = -0.5;
  b[2] =  3.7;
  b[3] = -0.4;

  expected = 0;
  expected += b[0] * b[0];
  expected += b[1] * b[1];
  expected += b[2] * b[2];
  expected += b[3] * b[3];
  ASSERT_FLOAT_EQ(expected, b.SquaredNorm());
}

TEST(Vector, Norm)
{
  float expected;

  ASSERT_FLOAT_EQ(sqrtf(2), Vector2f::Ones().Norm());
  ASSERT_FLOAT_EQ(sqrtf(3), Vector3f::Ones().Norm());
  ASSERT_FLOAT_EQ(sqrtf(4), Vector4f::Ones().Norm());

#ifndef NDEBUG
  ASSERT_THROW(Vector2f::Zeros().Norm(), Exception);
  ASSERT_THROW(Vector3f::Zeros().Norm(), Exception);
  ASSERT_THROW(Vector4f::Zeros().Norm(), Exception);
#endif

  Vector3f a;
  a[0] =  1.5;
  a[1] = -0.5;
  a[2] =  3.7;

  expected = 0;
  expected += a[0] * a[0];
  expected += a[1] * a[1];
  expected += a[2] * a[2];
  expected = sqrtf(expected);
  ASSERT_FLOAT_EQ(expected, a.Norm());

  Vector4f b;
  b[0] =  1.5;
  b[1] = -0.5;
  b[2] =  3.7;
  b[3] = -0.4;

  expected = 0;
  expected += b[0] * b[0];
  expected += b[1] * b[1];
  expected += b[2] * b[2];
  expected += b[3] * b[3];
  expected = sqrtf(expected);
  ASSERT_FLOAT_EQ(expected, b.Norm());
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
  Vector3f a;
  Vector3f found;
  Vector3f expected;

  a[0] =  1.5;
  a[1] = -0.5;
  a[2] =  3.7;

  norm = 0;
  norm += a[0] * a[0];
  norm += a[1] * a[1];
  norm += a[2] * a[2];
  norm = sqrtf(norm);

  expected[0] = a[0] / norm;
  expected[1] = a[1] / norm;
  expected[2] = a[2] / norm;

  found = a.Normalized();

  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);
  ASSERT_FLOAT_EQ(expected[2], found[2]);
  ASSERT_FLOAT_EQ(1, found.Norm());
}

TEST(Vector, Dot)
{
  Vector3f a;
  Vector3f b;
  float expected;

  a[0] =  1.5;
  a[1] = -0.5;
  a[2] =  3.7;

  b[0] = -0.7;
  b[1] = -2.1;
  b[2] =  4.9;

  expected = 0;
  expected += a[0] * b[0];
  expected += a[1] * b[1];
  expected += a[2] * b[2];

  ASSERT_FLOAT_EQ(expected, a.Dot(b));
  ASSERT_FLOAT_EQ(expected, b.Dot(a));
}

} // namespace testing

} // namespace vulcan