#include <gtest/gtest.h>
#include <vulcan/transform.h>

namespace vulcan
{
namespace testing
{

TEST(Transform, Constructor)
{
  Transform transform;
  const Matrix4f& matrix = transform.GetMatrix();
  const Matrix4f& inv_matrix = transform.GetInverseMatrix();

  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      const float expected = (i == j) ? 1 : 0;
      ASSERT_FLOAT_EQ(expected, matrix(j, i));
      ASSERT_FLOAT_EQ(expected, inv_matrix(j, i));
    }
  }
}

TEST(Tranform, Inverse)
{
  Transform a;
  Transform b;

  a = Transform::Translate(0.1, -0.3, 0.9) *
      Transform::Rotate(0.5925, 0.7040, -0.2221, 0.3226);

  b = a.Inverse();

  const Matrix4f& a_matrix = a.GetMatrix();
  const Matrix4f& b_matrix = b.GetMatrix();
  const Matrix4f& a_inv_matrix = a.GetInverseMatrix();
  const Matrix4f& b_inv_matrix = b.GetInverseMatrix();

  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      ASSERT_FLOAT_EQ(a_inv_matrix(j, i), b_matrix(j, i));
      ASSERT_FLOAT_EQ(a_matrix(j, i), b_inv_matrix(j, i));
    }
  }
}

TEST(Transform, RotateMatrix)
{
  Matrix4f matrix;
  Matrix4f inv_matrix;
  Transform transform;
  Matrix3f expected;
  Vector3f translation;

  expected(0, 0) =  0.6932;
  expected(1, 0) = -0.6950;
  expected(2, 0) =  0.1910;

  expected(0, 1) =  0.0696;
  expected(1, 1) = -0.1992;
  expected(2, 1) = -0.9775;

  expected(0, 2) =  0.7174;
  expected(1, 2) =  0.6909;
  expected(2, 2) = -0.0898;

  transform = Transform::Rotate(expected);
  inv_matrix = transform.GetInverseMatrix();
  matrix = transform.GetMatrix();

  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      ASSERT_NEAR(expected(j, i), matrix(j, i), 1E-4);
      ASSERT_NEAR(expected(j, i), inv_matrix(i, j), 1E-4);
    }
  }

  translation = transform.GetTranslation();
  ASSERT_EQ(0, translation[0]);
  ASSERT_EQ(0, translation[1]);
  ASSERT_EQ(0, translation[2]);

  // transform = Transform::Rotate(2 * expected);
  // inv_matrix = transform.GetInverseMatrix();
  // matrix = transform.GetMatrix();

  // for (int i = 0; i < 3; ++i)
  // {
  //   for (int j = 0; j < 3; ++j)
  //   {
  //     ASSERT_NEAR(expected(j, i), matrix(j, i), 1E-4);
  //     ASSERT_NEAR(expected(j, i), inv_matrix(i, j), 1E-4);
  //   }
  // }

  // translation = transform.GetTranslation();
  // ASSERT_EQ(0, translation[0]);
  // ASSERT_EQ(0, translation[1]);
  // ASSERT_EQ(0, translation[2]);

  // Matrix3f perturbed = expected;
  // perturbed(0, 1) *= 1.74f;
  // perturbed(1, 1) *= 1.74f;
  // perturbed(2, 1) *= 1.74f;
  // perturbed(2, 2) += 0.01f;

  // transform = Transform::Rotate(perturbed);
  // inv_matrix = transform.GetInverseMatrix();
  // matrix = transform.GetMatrix();

  // for (int i = 0; i < 3; ++i)
  // {
  //   for (int j = 0; j < 3; ++j)
  //   {
  //     ASSERT_NEAR(expected(j, i), matrix(j, i), 1E-4);
  //     ASSERT_NEAR(expected(j, i), inv_matrix(i, j), 1E-4);
  //   }
  // }

  // translation = transform.GetTranslation();
  // ASSERT_EQ(0, translation[0]);
  // ASSERT_EQ(0, translation[1]);
  // ASSERT_EQ(0, translation[2]);
}

TEST(Transform, Translate)
{
  Vector3f found;
  Vector3f expected;
  Transform transform;

  expected[0] =  1.1;
  expected[1] = -2.7;
  expected[2] =  3.9;

  transform = Transform::Translate(expected);
  found = transform.GetTranslation();

  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);
  ASSERT_FLOAT_EQ(expected[2], found[2]);

  found = Vector3f(transform * Vector4f(0, 0, 0, 1));

  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);
  ASSERT_FLOAT_EQ(expected[2], found[2]);

  transform = transform.Inverse();
  found = transform.GetTranslation();

  ASSERT_FLOAT_EQ(-expected[0], found[0]);
  ASSERT_FLOAT_EQ(-expected[1], found[1]);
  ASSERT_FLOAT_EQ(-expected[2], found[2]);

  found = Vector3f(transform * Vector4f(0, 0, 0, 1));

  ASSERT_FLOAT_EQ(-expected[0], found[0]);
  ASSERT_FLOAT_EQ(-expected[1], found[1]);
  ASSERT_FLOAT_EQ(-expected[2], found[2]);

  transform = Transform::Translate(expected[0], expected[1], expected[2]);
  found = transform.GetTranslation();

  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);
  ASSERT_FLOAT_EQ(expected[2], found[2]);

  found = Vector3f(transform * Vector4f(0, 0, 0, 1));

  ASSERT_FLOAT_EQ(expected[0], found[0]);
  ASSERT_FLOAT_EQ(expected[1], found[1]);
  ASSERT_FLOAT_EQ(expected[2], found[2]);

  transform = transform.Inverse();
  found = transform.GetTranslation();

  ASSERT_FLOAT_EQ(-expected[0], found[0]);
  ASSERT_FLOAT_EQ(-expected[1], found[1]);
  ASSERT_FLOAT_EQ(-expected[2], found[2]);

  found = Vector3f(transform * Vector4f(0, 0, 0, 1));

  ASSERT_FLOAT_EQ(-expected[0], found[0]);
  ASSERT_FLOAT_EQ(-expected[1], found[1]);
  ASSERT_FLOAT_EQ(-expected[2], found[2]);
}

} // namespace testing

} // namespace vulcan