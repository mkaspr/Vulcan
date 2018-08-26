#pragma once

#include <vulcan/device.h>
#include <vulcan/matrix.h>

namespace vulcan
{

class Transform
{
  public:

    VULCAN_HOST_DEVICE
    Transform() :
      matrix_(Matrix4f::Identity()),
      inv_matrix_(Matrix4f::Identity())
    {
    }

    VULCAN_HOST_DEVICE
    const Matrix4f& GetMatrix() const
    {
      return matrix_;
    }

    VULCAN_HOST_DEVICE
    const Matrix4f& GetInverseMatrix() const
    {
      return inv_matrix_;
    }

    VULCAN_HOST_DEVICE
    Vector3f GetTranslation() const
    {
      Vector3f result;
      result[0] = matrix_(0, 3);
      result[1] = matrix_(1, 3);
      result[2] = matrix_(2, 3);
      return result;
    }

    VULCAN_HOST_DEVICE
    inline Vector4f operator*(const Vector4f& p) const
    {
      Vector4f result;
      const Matrix4f& M = matrix_;
      const float x = p[0];
      const float y = p[1];
      const float z = p[2];
      const float w = p[3];

      result[0] = M(0, 0) * x + M(0, 1) * y + M(0, 2) * z + M(0, 3) * w;
      result[1] = M(1, 0) * x + M(1, 1) * y + M(1, 2) * z + M(1, 3) * w;
      result[2] = M(2, 0) * x + M(2, 1) * y + M(2, 2) * z + M(2, 3) * w;
      result[3] = w;

      return result;
    }

    VULCAN_HOST_DEVICE
    inline Transform operator*(const Transform& t) const
    {
      return Transform(matrix_ * t.matrix_, t.inv_matrix_ * inv_matrix_);
    }

    VULCAN_HOST_DEVICE
    inline Transform Inverse() const
    {
      return Transform(inv_matrix_, matrix_);
    }

    VULCAN_HOST_DEVICE
    static inline Transform Rotate(const Matrix3f& R)
    {
      Matrix4f matrix = Matrix4f::Identity();

      Vector3f x_axis(R(0, 0), R(1, 0), R(2, 0));
      Vector3f y_axis(R(0, 1), R(1, 1), R(2, 1));
      Vector3f z_axis(R(0, 2), R(1, 2), R(2, 2));

      // x_axis.Normalize();
      // y_axis.Normalize();
      // z_axis = x_axis.Cross(y_axis);
      // y_axis = z_axis.Cross(x_axis);

      matrix(0, 0) = x_axis[0];
      matrix(1, 0) = x_axis[1];
      matrix(2, 0) = x_axis[2];

      matrix(0, 1) = y_axis[0];
      matrix(1, 1) = y_axis[1];
      matrix(2, 1) = y_axis[2];

      matrix(0, 2) = z_axis[0];
      matrix(1, 2) = z_axis[1];
      matrix(2, 2) = z_axis[2];

      return Transform(matrix, matrix.Transpose());
    }

    VULCAN_HOST_DEVICE
    static inline Transform Rotate(const Vector4f& R)
    {
      return Rotate(R[0], R[1], R[2], R[3]);
    }

    VULCAN_HOST_DEVICE
    static inline Transform Rotate(float w, float x, float y, float z)
    {
      Matrix4f matrix;

      matrix(0, 0) = 1 - 2 * (y * y + z * z);
      matrix(0, 1) = 2 * (x * y - w * z);
      matrix(0, 2) = 2 * (x * z + w * y);
      matrix(0, 3) = 0.0f;

      matrix(1, 0) = 2 * (x * y + w * z);
      matrix(1, 1) = 1 - 2 * (x * x + z * z);
      matrix(1, 2) = 2 * (y * z - w * x);
      matrix(1, 3) = 0.0f;

      matrix(2, 0) = 2 * (x * z - w * y);
      matrix(2, 1) = 2 * (y * z + w * x);
      matrix(2, 2) = 1 - 2 * (x * x + y * y);
      matrix(2, 3) = 0.0f;

      matrix(3, 0) = 0.0f;
      matrix(3, 1) = 0.0f;
      matrix(3, 2) = 0.0f;
      matrix(3, 3) = 1.0f;

      return Transform(matrix, matrix.Transpose());
    }

    VULCAN_HOST_DEVICE
    static inline Transform Translate(const Vector3f& t)
    {
      return Translate(t[0], t[1], t[2]);
    }

    VULCAN_HOST_DEVICE
    static inline Transform Translate(float x, float y, float z)
    {
      Matrix4f matrix = Matrix4f::Identity();
      matrix(0, 3) = x;
      matrix(1, 3) = y;
      matrix(2, 3) = z;

      Matrix4f inv_matrix = Matrix4f::Identity();
      inv_matrix(0, 3) = -x;
      inv_matrix(1, 3) = -y;
      inv_matrix(2, 3) = -z;

      return Transform(matrix, inv_matrix);
    }

  protected:

    VULCAN_HOST_DEVICE
    Transform(const Matrix4f& matrix, const Matrix4f& inv_matrix) :
      matrix_(matrix),
      inv_matrix_(inv_matrix)
    {
    }

  protected:

    Matrix4f matrix_;

    Matrix4f inv_matrix_;
};

} // namespace vulcan