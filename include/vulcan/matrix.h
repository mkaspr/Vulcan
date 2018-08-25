#pragma once

#include <iomanip>
#include <iostream>
#include <vulcan/device.h>
#include <vulcan/exception.h>
#include <vulcan/math.h>

namespace vulcan
{

template <typename T, int M, int N>
class Matrix
{
  static_assert(M > 0 && N > 0, "invalid matrix size");

  public:

    VULCAN_HOST_DEVICE
    Matrix()
    {
    }

    VULCAN_HOST_DEVICE
    Matrix(T v0, T v1)
    {
      static_assert(M * N == 2, "2D vector required");
      data_[0] = v0;
      data_[1] = v1;
    }

    VULCAN_HOST_DEVICE
    Matrix(T v0, T v1, T v2)
    {
      static_assert(M * N == 3, "3D vector required");
      data_[0] = v0;
      data_[1] = v1;
      data_[2] = v2;
    }

    VULCAN_HOST_DEVICE
    Matrix(T v0, T v1, T v2, T v3)
    {
      static_assert(M * N == 4 && (M == 1 || N == 1), "4D vector required");
      data_[0] = v0;
      data_[1] = v1;
      data_[2] = v2;
      data_[3] = v3;
    }

    template <typename U>
    VULCAN_HOST_DEVICE
    explicit Matrix(const Matrix<U, M, N>& matrix)
    {
      for (int i = 0; i < N; ++i)
      {
        for (int j = 0; j < M; ++j)
        {
          (*this)(j, i) = matrix(j, i);
        }
      }
    }

    template <int P, int Q>
    VULCAN_HOST_DEVICE
    explicit Matrix(const Matrix<T, P, Q>& matrix)
    {
      static_assert(M == 1 || N == 1, "vector required");
      static_assert(P == 1 || Q == 1, "vector argument required");
      const int limit = min(M * N, P * Q);

      for (int i = 0; i < limit; ++i)
      {
        data_[i] = matrix[i];
      }

      for (int i = limit; i < M * N; ++i)
      {
        data_[i] = T(0);
      }
    }

    template <int P, int Q>
    VULCAN_HOST_DEVICE
    explicit Matrix(const Matrix<T, P, Q>& matrix, T v0)
    {
      static_assert(M == 1 || N == 1, "vector required");
      static_assert(P == 1 || Q == 1, "vector argument required");
      static_assert(M * N == P * Q + 1, "invalid vector length");

      for (int i = 0; i < P * Q; ++i)
      {
        data_[i] = matrix[i];
      }

      data_[M * N - 1] = v0;
    }

    VULCAN_HOST_DEVICE
    inline int GetRows() const
    {
      return M;
    }

    VULCAN_HOST_DEVICE
    inline int GetColumns() const
    {
      return N;
    }

    VULCAN_HOST_DEVICE
    inline int GetTotal() const
    {
      return M * N;
    }

    VULCAN_HOST_DEVICE
    inline T Norm() const
    {
      const float squared_normal = SquaredNorm();
      VULCAN_DEBUG(squared_normal > 0);
      return sqrt(squared_normal);
    }

    VULCAN_HOST_DEVICE
    inline T SquaredNorm() const
    {
      return this->Dot(*this);
    }

    VULCAN_HOST_DEVICE
    inline void Normalize()
    {
      const T inv_norm = T(1) / Norm();

      for (int i = 0; i < M * N; ++i)
      {
        data_[i] *= inv_norm;
      }
    }

    VULCAN_HOST_DEVICE
    inline Matrix Normalized() const
    {
      Matrix result(*this);
      const T inv_norm = T(1) / Norm();

      for (int i = 0; i < M * N; ++i)
      {
        result.data_[i] *= inv_norm;
      }

      return result;
    }

    VULCAN_HOST_DEVICE
    inline float Dot(const Matrix& rhs) const
    {
      static_assert(M == 1 || N == 1, "vector required");

      float result = 0;

      for (int i = 0; i < M * N; ++i)
      {
        result += data_[i] * rhs.data_[i];
      }

      return result;
    }

    VULCAN_HOST_DEVICE
    inline Matrix Cross(const Matrix& rhs) const
    {
      static_assert(M * N == 3, "3D vector required");

      Matrix result;
      result[0] = (data_[1] * rhs[2]) - (data_[2] * rhs[1]);
      result[1] = (data_[2] * rhs[0]) - (data_[0] * rhs[2]);
      result[2] = (data_[0] * rhs[1]) - (data_[1] * rhs[0]);
      return result;
    }

    VULCAN_HOST_DEVICE
    inline Matrix<T, N, M> Transpose() const
    {
      Matrix<T, N, M> result;

      for (int n = 0; n < N; ++n)
      {
        for (int m = 0; m < M; ++m)
        {
          result(n, m) = (*this)(m, n);
        }
      }

      return result;
    }

    VULCAN_HOST_DEVICE
    const Matrix operator+(const Matrix& matrix) const
    {
      Matrix result(*this);
      result += matrix;
      return result;
    }

    VULCAN_HOST_DEVICE
    Matrix& operator+=(const Matrix& matrix)
    {
      for (int i = 0; i < M * N; ++i)
      {
        data_[i] += matrix.data_[i];
      }

      return *this;
    }

    VULCAN_HOST_DEVICE
    const Matrix operator-(const Matrix& matrix) const
    {
      Matrix result(*this);
      result -= matrix;
      return result;
    }

    VULCAN_HOST_DEVICE
    Matrix& operator-=(const Matrix& matrix)
    {
      for (int i = 0; i < M * N; ++i)
      {
        data_[i] -= matrix.data_[i];
      }

      return *this;
    }

    template <typename S>
    VULCAN_HOST_DEVICE
    const Matrix operator+(S scalar) const
    {
      Matrix result(*this);
      result += scalar;
      return result;
    }

    template <typename S>
    VULCAN_HOST_DEVICE
    Matrix& operator+=(S scalar)
    {
      for (int i = 0; i < M * N; ++i)
      {
        data_[i] += scalar;
      }

      return *this;
    }

    template <typename S>
    VULCAN_HOST_DEVICE
    const Matrix operator*(S scalar) const
    {
      Matrix result(*this);
      result *= scalar;
      return result;
    }

    template <typename S>
    VULCAN_HOST_DEVICE
    Matrix& operator*=(S scalar)
    {
      for (int i = 0; i < M * N; ++i)
      {
        data_[i] *= scalar;
      }

      return *this;
    }

    template <typename S>
    VULCAN_HOST_DEVICE
    const Matrix operator/(S scalar) const
    {
      Matrix result(*this);
      result /= scalar;
      return result;
    }

    template <typename S>
    VULCAN_HOST_DEVICE
    Matrix& operator/=(S scalar)
    {
      VULCAN_DEBUG(scalar != S(0));
      const float inv = 1.0f / scalar;
      return (*this) *= inv;
    }

    template <int P>
    VULCAN_HOST_DEVICE
    const Matrix<T, M, P> operator*(const Matrix<T, N, P>& matrix) const
    {
      Matrix<T, M, P> result;

      for (int p = 0; p < P; ++p)
      {
        for (int m = 0; m < M; ++m)
        {
          result(m, p) = 0;

          for (int n = 0; n < N; ++n)
          {
            result(m, p) += (*this)(m, n) * matrix(n, p);
          }
        }
      }

      return result;
    }

    VULCAN_HOST_DEVICE
    const T& operator()(int row, int col) const
    {
      VULCAN_DEBUG_MSG(row >= 0 && row < M, "index out of bounds");
      VULCAN_DEBUG_MSG(col >= 0 && col < N, "index out of bounds");
      return data_[col * M + row];
    }

    VULCAN_HOST_DEVICE
    T& operator()(int row, int col)
    {
      VULCAN_DEBUG_MSG(row >= 0 && row < M, "index out of bounds");
      VULCAN_DEBUG_MSG(col >= 0 && col < N, "index out of bounds");
      return data_[col * M + row];
    }

    VULCAN_HOST_DEVICE
    const T& operator[](int index) const
    {
      VULCAN_DEBUG_MSG(index >= 0 && index < M * N, "index out of bounds");
      static_assert(M == 1 || N == 1, "vector required");
      return data_[index];
    }

    VULCAN_HOST_DEVICE
    T& operator[](int index)
    {
      VULCAN_DEBUG_MSG(index >= 0 && index < M * N, "index out of bounds");
      static_assert(M == 1 || N == 1, "vector required");
      return data_[index];
    }

    VULCAN_HOST_DEVICE
    inline bool operator==(const Matrix& matrix) const
    {
      for (int i = 0; i < M * N; ++i)
      {
        if (data_[i] != matrix.data_[i]) return false;
      }

      return true;
    }

    VULCAN_HOST_DEVICE
    static Matrix Zeros()
    {
      Matrix result;

      for (int i = 0; i < M * N; ++i)
      {
        result.data_[i] = T(0);
      }

      return result;
    }

    VULCAN_HOST_DEVICE
    static Matrix Ones()
    {
      Matrix result;

      for (int i = 0; i < M * N; ++i)
      {
        result.data_[i] = T(1);
      }

      return result;
    }

    VULCAN_HOST_DEVICE
    static Matrix Identity()
    {
      static_assert(M == N, "square matrix required");
      Matrix result = Matrix::Zeros();

      for (int i = 0; i < M; ++i)
      {
        result.data_[i * (M + 1)] = T(1);
      }

      return result;
    }

  protected:

    T data_[M * N];
};

template <typename S, typename T, int M, int N>
VULCAN_HOST_DEVICE
inline Matrix<T, M, N> operator+(S scalar, const Matrix<T, M, N>& matrix)
{
  return matrix + scalar;
}

template <typename T, int M, int N>
VULCAN_HOST_DEVICE
inline Matrix<T, M, N> operator*(float scalar, const Matrix<T, M, N>& matrix)
{
  return matrix * scalar;
}

template <typename T, int M, int N>
VULCAN_HOST
std::ostream& operator<<(std::ostream& out, const Matrix<T, M, N>& matrix)
{
  for (int m = 0; m < M - 1; ++m)
  {
    for (int n = 0; n < N - 1; ++n)
    {
      out << std::setw(9) << matrix(m, n) << " ";
    }

    out << std::setw(9) << matrix(m, N - 1) << std::endl;
  }

  for (int n = 0; n < N - 1; ++n)
  {
    out << std::setw(9) << matrix(M - 1, n) << " ";
  }

  return out << std::setw(9) << matrix(M - 1, N - 1);
}

template <typename T, int M>
using Vector = Matrix<T, M, 1>;

typedef Vector<char, 2> Vector2c;
typedef Vector<char, 3> Vector3c;
typedef Vector<char, 4> Vector4c;
typedef Vector<int, 2> Vector2i;
typedef Vector<int, 3> Vector3i;
typedef Vector<int, 4> Vector4i;
typedef Vector<short, 2> Vector2s;
typedef Vector<short, 3> Vector3s;
typedef Vector<short, 4> Vector4s;
typedef Vector<float, 2> Vector2f;
typedef Vector<float, 3> Vector3f;
typedef Vector<float, 4> Vector4f;
typedef Vector<float, 6> Vector6f;
typedef Matrix<float, 2, 2> Matrix2f;
typedef Matrix<float, 3, 3> Matrix3f;
typedef Matrix<float, 4, 4> Matrix4f;

} // namespace vulcan