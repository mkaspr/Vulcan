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

    Tracker(std::shared_ptr<const Volume> volume);

    virtual ~Tracker();

    std::shared_ptr<const Volume> GetVolume() const;

    std::shared_ptr<const Frame> GetKeyframe() const;

    void SetKeyframe(std::shared_ptr<const Frame> keyframe);

    void Track(Frame& frame);

  protected:

    bool IsSolving() const;

    bool HasHaulted() const;

    bool HasConverged() const;

    double GetCostChange() const;

    double GetMinCostChange() const;

    void BeginSolve();

    void CreateState();

    void ValidateKeyframe();

    void UpdateSolve();

    void ComputeOperands();

    void ComputeResidual();

    void ComputeJacobian();

    void ComputeHessian();

    void ComputeGradient();

    void ComputeUpdate();

    void UpdateState();

    void UpdateCost();

    void EndSolve();

  private:

    void Initialize();

    void CreateHandle();

    void CreateResidualBuffer();

    void CreateJacobianBuffer();

    void CreateHessianBuffer();

    void CreateGradientBuffer();

  protected:

    int iteration_;

    int max_iterations_;

    double curr_cost_;

    double prev_cost_;

    double change_threshold_;

    std::shared_ptr<const Volume> volume_;

    std::shared_ptr<const Frame> keyframe_;

    Buffer<double> residuals_;

    Buffer<double> jacobian_;

    Buffer<double> hessian_;

    Buffer<double> gradient_;

    cublasHandle_t handle_;
};

} // namespace vulcan