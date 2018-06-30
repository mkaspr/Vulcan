#include <gtest/gtest.h>
#include <vulcan/projection.h>

namespace vulcan
{
namespace testing
{

TEST(Projection, Constructor)
{
  Projection projection;
  ASSERT_FLOAT_EQ(320, projection.GetFocalLength()[0]);
  ASSERT_FLOAT_EQ(320, projection.GetFocalLength()[1]);
  ASSERT_FLOAT_EQ(320, projection.GetCenterPoint()[0]);
  ASSERT_FLOAT_EQ(240, projection.GetCenterPoint()[1]);
}

TEST(Projection, FocalLength)
{
  Projection projection;

  projection.SetFocalLength(320, 240);
  ASSERT_EQ(320, projection.GetFocalLength()[0]);
  ASSERT_EQ(240, projection.GetFocalLength()[1]);

  projection.SetFocalLength(Vector2f(160, 120));
  ASSERT_EQ(160, projection.GetFocalLength()[0]);
  ASSERT_EQ(120, projection.GetFocalLength()[1]);

#ifndef NDEBUG
  const float nan = std::numeric_limits<float>::quiet_NaN();
  ASSERT_THROW(projection.SetFocalLength(0, 32), Exception);
  ASSERT_THROW(projection.SetFocalLength(Vector2f(32, 0)), Exception);
  ASSERT_THROW(projection.SetFocalLength(nan, 32), Exception);
  ASSERT_THROW(projection.SetFocalLength(Vector2f(32, nan)), Exception);
#endif
}

TEST(Projection, CenterPoint)
{
  Projection projection;

  projection.SetCenterPoint(320, 240);
  ASSERT_EQ(320, projection.GetCenterPoint()[0]);
  ASSERT_EQ(240, projection.GetCenterPoint()[1]);

  projection.SetCenterPoint(Vector2f(160, 120));
  ASSERT_EQ(160, projection.GetCenterPoint()[0]);
  ASSERT_EQ(120, projection.GetCenterPoint()[1]);

#ifndef NDEBUG
  const float nan = std::numeric_limits<float>::quiet_NaN();
  ASSERT_THROW(projection.SetCenterPoint(nan, 32), Exception);
  ASSERT_THROW(projection.SetCenterPoint(Vector2f(32, nan)), Exception);
#endif
}

TEST(Projection, Project)
{
  Projection projection;
  Vector2f found, expected;
  Vector3f point;
  Matrix3f K;
  Vector3f uvw;

  projection.SetFocalLength(325, 315);
  projection.SetCenterPoint(325, 235);

  K = Matrix3f::Identity();
  K(0, 0) = projection.GetFocalLength()[0];
  K(1, 1) = projection.GetFocalLength()[1];
  K(0, 2) = projection.GetCenterPoint()[0];
  K(1, 2) = projection.GetCenterPoint()[1];

  point = Vector3f(0, 0, 1);
  expected = Vector2f(325, 235);

  found = projection.Project(point[0], point[1], point[2]);
  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);

  found = projection.Project(point);
  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);

  point = Vector3f(23.4, -0.725, 31.2);
  uvw = K * point;
  expected[0] = uvw[0] / uvw[2];
  expected[1] = uvw[1] / uvw[2];

  found = projection.Project(point[0], point[1], point[2]);
  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);

  found = projection.Project(point);
  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);
}

TEST(Projection, Unproject)
{
  Projection projection;
  Vector3f found, expected;
  Matrix3f Kinv;
  Vector2f uv;

  projection.SetFocalLength(325, 315);
  projection.SetCenterPoint(325, 235);

  Kinv = Matrix3f::Identity();
  Kinv(0, 0) = 1.0f / projection.GetFocalLength()[0];
  Kinv(1, 1) = 1.0f / projection.GetFocalLength()[1];
  Kinv(0, 2) = -projection.GetCenterPoint()[0] * Kinv(0, 0);
  Kinv(1, 2) = -projection.GetCenterPoint()[1] * Kinv(1, 1);

  uv = projection.GetCenterPoint();
  expected = Vector3f(0, 0, 1);

  found = projection.Unproject(uv[0], uv[1]);
  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);
  ASSERT_FLOAT_EQ(expected[2], found[2]);

  found = projection.Unproject(uv);
  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);
  ASSERT_FLOAT_EQ(expected[2], found[2]);

  uv = Vector2f(42.52, 230.71);
  expected = Kinv * Vector3f(uv, 1);

  found = projection.Unproject(uv[0], uv[1]);
  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);
  ASSERT_FLOAT_EQ(expected[2], found[2]);

  found = projection.Unproject(uv);
  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);
  ASSERT_FLOAT_EQ(expected[2], found[2]);
}

} // namespace testing

} // namespace vulcan