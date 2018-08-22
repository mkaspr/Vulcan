#pragma once

#include <memory>
#include <vulcan/buffer.h>
#include <vulcan/device.h>

namespace vulcan
{

class Frame;
class Volume;

class Tracker
{
  public:

    Tracker();

    virtual ~Tracker();

    std::shared_ptr<const Frame> GetKeyframe() const;

    void SetKeyframe(std::shared_ptr<const Frame> keyframe);

    bool GetTranslationEnabled() const;

    void SetTranslationEnabled(bool enabled);

    int GetMaxIterations() const;

    void SetMaxIterations(int iterations);

    void Track(Frame& frame);

  protected:

    bool IsSolving() const;

    virtual void BeginSolve(const Frame& frame);

    void CreateState(const Frame& frame);

    void ValidateKeyframe() const;

    void ValidateFrame(const Frame& frame) const;

    void UpdateSolve(Frame& frame);

    void ComputeOperands(const Frame& frame);

    virtual void ComputeResidual(const Frame& frame) = 0;

    virtual void ComputeJacobian(const Frame& frame) = 0;

    void ComputeHessian();

    void ComputeGradient();

    void ComputeUpdate(Frame& frame);

    void UpdateState(const Frame& frame);

    void EndSolve();

    void ResizeBuffers(const Frame& frame);

    void ResizeResidualBuffer(const Frame& frame);

    void ResizeJacobianBuffer(const Frame& frame);

    void ResizeHessianBuffer();

    void ResizeGradientBuffer();

    virtual int GetResidualCount(const Frame& frame) const = 0;

    int GetParameterCount() const;

  private:

    void Initialize();

  protected:

    bool translation_enabled_;

    int iteration_;

    int max_iterations_;

    std::shared_ptr<const Frame> keyframe_;

    Buffer<float> residuals_;

    Buffer<float> jacobian_;

    Buffer<float> hessian_;

    Buffer<float> gradient_;

    cublasHandle_t handle_;
};

} // namespace vulcan