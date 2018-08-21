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

    std::shared_ptr<const Volume> GetVolume() const;

    std::shared_ptr<const Frame> GetKeyframe() const;

    void SetKeyframe(std::shared_ptr<const Frame> keyframe);

    bool GetTranslationEnabled() const;

    void SetTranslationEnabled(bool enabled);

    void Track(Frame& frame);

  protected:

    bool IsSolving() const;

    void BeginSolve(const Frame& frame);

    void CreateState(const Frame& frame);

    void ValidateKeyframe() const;

    void ValidateFrame(const Frame& frame) const;

    void UpdateSolve(Frame& frame);

    void ComputeOperands(const Frame& frame);

    void ComputeResidual(const Frame& frame);

    void ComputeJacobian(const Frame& frame);

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

    int GetResidualCount(const Frame& frame) const;

    int GetParameterCount() const;

  private:

    void Initialize();

  protected:

    bool translation_enabled_;

    int iteration_;

    int max_iterations_;

    std::shared_ptr<const Frame> keyframe_;

    Buffer<double> residuals_;

    Buffer<double> jacobian_;

    Buffer<double> hessian_;

    Buffer<double> gradient_;

    cublasHandle_t handle_;
};

} // namespace vulcan