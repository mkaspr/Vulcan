#pragma once

#include <vulcan/math.h>
#include <vulcan/matrix.h>
#include <vulcan/device.h>
#include <vulcan/exception.h>

namespace vulcan
{

class Projection
{
  public:

    VULCAN_HOST_DEVICE
    Projection() :
      focal_length_(500, 500),
      center_point_(320, 240)
    {
    }

    VULCAN_HOST_DEVICE
    inline const Vector2f& GetFocalLength() const
    {
      return focal_length_;
    }

    VULCAN_HOST_DEVICE
    inline void SetFocalLength(const Vector2f& length)
    {
      VULCAN_DEBUG(length[0] > 0 && length[1] > 0);
      VULCAN_DEBUG(!isnan(length[0]));
      VULCAN_DEBUG(!isnan(length[1]));
      focal_length_ = length;
    }

    VULCAN_HOST_DEVICE
    inline void SetFocalLength(float w, float h)
    {
      SetFocalLength(Vector2f(w, h));
    }

    VULCAN_HOST_DEVICE
    inline const Vector2f& GetCenterPoint() const
    {
      return center_point_;
    }

    VULCAN_HOST_DEVICE
    inline void SetCenterPoint(const Vector2f& point)
    {
      VULCAN_DEBUG(!isnan(point[0]));
      VULCAN_DEBUG(!isnan(point[1]));
      center_point_ = point;
    }

    VULCAN_HOST_DEVICE
    inline void SetCenterPoint(float w, float h)
    {
      SetCenterPoint(Vector2f(w, h));
    }

    VULCAN_HOST_DEVICE
    inline Vector2f Project(const Vector3f& Xcp) const
    {
      Vector2f result;
      const float inv_w = 1.0f / Xcp[2];
      result[0] = inv_w * focal_length_[0] * Xcp[0] + center_point_[0];
      result[1] = inv_w * focal_length_[1] * Xcp[1] + center_point_[1];
      return result;
    }

    VULCAN_HOST_DEVICE
    inline Vector2f Project(float x, float y, float z) const
    {
      return Project(Vector3f(x, y, z));
    }

    VULCAN_HOST_DEVICE
    inline Vector3f Unproject(const Vector2f& uv) const
    {
      Vector3f result;
      const float ifx = 1.0f / focal_length_[0];
      const float ify = 1.0f / focal_length_[1];
      result[0] = ifx * uv[0] - center_point_[0] * ifx;
      result[1] = ify * uv[1] - center_point_[1] * ify;
      result[2] = 1;
      return result;
    }

    VULCAN_HOST_DEVICE
    inline Vector3f Unproject(float u, float v) const
    {
      return Unproject(Vector2f(u, v));
    }

    VULCAN_HOST_DEVICE
    inline Vector3f Unproject(const Vector2f& uv, float d) const
    {
      return d * Unproject(uv);
    }

    VULCAN_HOST_DEVICE
    inline Vector3f Unproject(float u, float v, float d) const
    {
      return Unproject(Vector2f(u, v), d);
    }

  protected:

    Vector2f focal_length_;

    Vector2f center_point_;
};

} // namespace vulcan