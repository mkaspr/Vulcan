#include <vulcan/depth_tracker.h>
#include <vulcan/device.h>
#include <vulcan/frame.h>
#include <vulcan/projection.h>
#include <vulcan/transform.h>

namespace vulcan
{
namespace
{

VULCAN_GLOBAL
void ComputeResidualKernel(const Transform Tmc, const float* keyframe_depths,
    const Vector3f* keyframe_normals, const Projection keyframe_projection,
    int keyframe_width, int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const Projection frame_projection,
    int frame_width, int frame_height, float* residuals)
{
  const int frame_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int frame_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (frame_x < frame_width && frame_y < frame_height)
  {
    float residual = 0;
    const int frame_index = frame_y * frame_width + frame_x;
    const float frame_depth = frame_depths[frame_index];

    if (frame_depth > 0)
    {
      const float frame_u = frame_x + 0.5f;
      const float frame_v = frame_y + 0.5f;
      const Vector3f Xcp = frame_projection.Unproject(frame_u, frame_v, frame_depth);
      const Vector3f Xmp = Vector3f(Tmc * Vector4f(Xcp, 1));
      const Vector2f keyframe_uv = keyframe_projection.Project(Xmp);

      if (keyframe_uv[0] >= 0 && keyframe_uv[0] < keyframe_width &&
          keyframe_uv[1] >= 0 && keyframe_uv[1] < keyframe_height)
      {
        const int keyframe_x = keyframe_uv[0];
        const int keyframe_y = keyframe_uv[1];
        const int keyframe_index = keyframe_y * keyframe_width + keyframe_x;
        const float keyframe_depth = keyframe_depths[keyframe_index];

        if (keyframe_depth > 0)
        {
          const Vector3f frame_normal = frame_normals[frame_index];
          const Vector3f keyframe_normal = keyframe_normals[keyframe_index];

          if (keyframe_normal.SquaredNorm() > 0 &&
              frame_normal.Dot(keyframe_normal) > 0.5f)
          {
            const Vector3f Ymp = keyframe_projection.Unproject(keyframe_uv, keyframe_depth);
            const Vector3f delta = Xmp - Ymp;

            if (delta.SquaredNorm() < 0.05)
            {
              residual = delta.Dot(keyframe_normal);
            }
          }
        }
      }
    }

    residuals[frame_index] = residual;
  }
}

template <bool translation_enabled>
VULCAN_GLOBAL
void ComputeJacobianKernel(const Transform Tmc, const float* keyframe_depths,
    const Vector3f* keyframe_normals, const Projection keyframe_projection,
    int keyframe_width, int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const Projection frame_projection,
    int frame_width, int frame_height, float* jacobian)
{
  const int frame_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int frame_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (frame_x < frame_width && frame_y < frame_height)
  {
    Vector6f dfdx = Vector6f::Zeros();
    const int frame_index = frame_y * frame_width + frame_x;
    const float frame_depth = frame_depths[frame_index];

    if (frame_depth > 0)
    {
      const float frame_u = frame_x + 0.5f;
      const float frame_v = frame_y + 0.5f;
      const Vector3f Xcp = frame_projection.Unproject(frame_u, frame_v, frame_depth);
      const Vector3f Xmp = Vector3f(Tmc * Vector4f(Xcp, 1));
      const Vector2f keyframe_uv = keyframe_projection.Project(Xmp);

      if (keyframe_uv[0] >= 0 && keyframe_uv[0] < keyframe_width &&
          keyframe_uv[1] >= 0 && keyframe_uv[1] < keyframe_height)
      {
        const int keyframe_x = keyframe_uv[0];
        const int keyframe_y = keyframe_uv[1];
        const int keyframe_index = keyframe_y * keyframe_width + keyframe_x;
        const float keyframe_depth = keyframe_depths[keyframe_index];

        if (keyframe_depth > 0)
        {
          const Vector3f frame_normal = frame_normals[frame_index];
          const Vector3f keyframe_normal = keyframe_normals[keyframe_index];

          if (keyframe_normal.SquaredNorm() > 0 &&
              frame_normal.Dot(keyframe_normal) > 0.5f)
          {
            const Vector3f Ymp = keyframe_projection.Unproject(keyframe_uv, keyframe_depth);
            const Vector3f delta = Xmp - Ymp;

            if (delta.SquaredNorm() < 0.05)
            {
              dfdx[0] = keyframe_normal[2] * Xcp[1] - keyframe_normal[1] * Xcp[2];
              dfdx[1] = keyframe_normal[0] * Xcp[2] - keyframe_normal[2] * Xcp[0];
              dfdx[2] = keyframe_normal[1] * Xcp[0] - keyframe_normal[0] * Xcp[1];

              if (translation_enabled)
              {
                dfdx[3] = keyframe_normal[0];
                dfdx[4] = keyframe_normal[1];
                dfdx[5] = keyframe_normal[2];
              }
            }
          }
        }
      }
    }

    const int residual_count = frame_width * frame_height;
    jacobian[0 * residual_count + frame_index] = dfdx[0];
    jacobian[1 * residual_count + frame_index] = dfdx[1];
    jacobian[2 * residual_count + frame_index] = dfdx[2];

    if (translation_enabled)
    {
      jacobian[3 * residual_count + frame_index] = dfdx[3];
      jacobian[4 * residual_count + frame_index] = dfdx[4];
      jacobian[5 * residual_count + frame_index] = dfdx[5];
    }
  }
}

} // namespace

void DepthTracker::ComputeResidual(const Frame& frame)
{
  const int frame_width = frame.depth_image->GetWidth();
  const int frame_height = frame.depth_image->GetHeight();
  const int keyframe_width = keyframe_->depth_image->GetWidth();
  const int keyframe_height = keyframe_->depth_image->GetHeight();
  const float* frame_depths = frame.depth_image->GetData();
  const float* keyframe_depths = keyframe_->depth_image->GetData();
  const Vector3f* keyframe_normals = keyframe_->normal_image->GetData();
  const Vector3f* frame_normals = frame.normal_image->GetData();
  const Projection& frame_projection = frame.projection;
  const Projection& keyframe_projection = keyframe_->projection;
  const Transform Tmc = keyframe_->Tcw * frame.Tcw.Inverse();
  float* residuals = residuals_.GetData();

  const dim3 threads(16, 16);
  const dim3 total(frame_width, frame_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(ComputeResidualKernel, blocks, threads, 0, 0, Tmc,
      keyframe_depths, keyframe_normals, keyframe_projection, keyframe_width,
      keyframe_height, frame_depths, frame_normals, frame_projection,
      frame_width, frame_height, residuals);
}

void DepthTracker::ComputeJacobian(const Frame& frame)
{
  const int frame_width = frame.depth_image->GetWidth();
  const int frame_height = frame.depth_image->GetHeight();
  const int keyframe_width = keyframe_->depth_image->GetWidth();
  const int keyframe_height = keyframe_->depth_image->GetHeight();
  const float* frame_depths = frame.depth_image->GetData();
  const float* keyframe_depths = keyframe_->depth_image->GetData();
  const Vector3f* keyframe_normals = keyframe_->normal_image->GetData();
  const Vector3f* frame_normals = frame.normal_image->GetData();
  const Projection& frame_projection = frame.projection;
  const Projection& keyframe_projection = keyframe_->projection;
  const Transform Tmc = keyframe_->Tcw * frame.Tcw.Inverse();
  float* jacobian = jacobian_.GetData();

  const dim3 threads(16, 16);
  const dim3 total(frame_width, frame_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  if (translation_enabled_)
  {
    CUDA_LAUNCH(ComputeJacobianKernel<true>, blocks, threads, 0, 0, Tmc,
        keyframe_depths, keyframe_normals, keyframe_projection, keyframe_width,
        keyframe_height, frame_depths, frame_normals, frame_projection,
        frame_width, frame_height, jacobian);
  }
  else
  {
    CUDA_LAUNCH(ComputeJacobianKernel<false>, blocks, threads, 0, 0, Tmc,
        keyframe_depths, keyframe_normals, keyframe_projection, keyframe_width,
        keyframe_height, frame_depths, frame_normals, frame_projection,
        frame_width, frame_height, jacobian);
  }
}

} // namespace vulcan