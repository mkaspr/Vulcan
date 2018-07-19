#pragma once

#include <cublas_v2.h>
#include <vulcan/buffer.h>
#include <vulcan/matrix.h>

namespace vulcan
{

class Detector
{
  public:

    Detector();

    virtual ~Detector();

    float GetRadius() const;

    void SetRadius(float radius);

    const Vector3f& GetOrigin() const;

    void SetOrigin(const Vector3f& origin);

    const Vector2f& GetBounds(int axis) const;

    void SetBounds(int axis, const Vector2f& bounds);

    int GetMinInlierCount() const;

    void SetMinInlierCount(int count);

    Vector3f Detect(const Buffer<Vector3f>& points);

  protected:

    bool BoxDetected() const;

    Vector3f GetValidPosition() const;

    Vector3f GetInvalidPosition() const;

    void Filter(const Buffer<Vector3f>& points);

    int GetBufferSize() const;

    void ResetBufferSize() const;

  private:

    void Initialize();

    void CreateHandle();

    void CreateBounds();

  protected:

    float radius_;

    Vector3f origin_;

    Vector2f bounds_[3];

    Buffer<Vector3f> points_;

    Buffer<float> distances_;

    int min_inlier_count_;

    cublasHandle_t handle_;
};

} // namespace vulcan