#include <vulcan/depth_tracker.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <vulcan/device.h>
#include <vulcan/frame.h>
#include <vulcan/projection.h>
#include <vulcan/transform.h>
#include <vulcan/util.cuh>

namespace vulcan
{

namespace
{

template <bool translation_enabled>
VULCAN_DEVICE
inline void Evaluate(int frame_x, int frame_y, const Transform& Tmc,
    const float* keyframe_depths, const Vector3f* keyframe_normals,
    const Projection& keyframe_projection, int keyframe_width,
    int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const Projection& frame_projection,
    int frame_width, int frame_height, float* residual, Vector6f* jacobian)
{
  VULCAN_DEBUG(frame_y >= 0 && frame_x < frame_width);
  VULCAN_DEBUG(frame_y >= 0 && frame_y < frame_height);

  if (residual) *residual = 0;
  if (jacobian) *jacobian = Vector6f::Zeros();

  if (frame_x < frame_width && frame_y < frame_height)
  {
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
          Vector3f frame_normal = frame_normals[frame_index];
          frame_normal = Vector3f(Tmc * Vector4f(frame_normal, 0));
          const Vector3f keyframe_normal = keyframe_normals[keyframe_index];

          if (keyframe_normal.SquaredNorm() > 0 &&
              frame_normal.Dot(keyframe_normal) > 0.5f)
          {
            Vector2f final_keyframe_uv;
            final_keyframe_uv[0] = floorf(keyframe_uv[0]) + 0.5f;
            final_keyframe_uv[1] = floorf(keyframe_uv[1]) + 0.5f;
            const Vector3f Ymp = keyframe_projection.Unproject(final_keyframe_uv, keyframe_depth);
            const Vector3f delta = Xmp - Ymp;

            if (delta.SquaredNorm() < 0.05)
            {
              if (residual) *residual = delta.Dot(keyframe_normal);

              if (jacobian)
              {
                (*jacobian)[0] = keyframe_normal[2] * Xcp[1] - keyframe_normal[1] * Xcp[2];
                (*jacobian)[1] = keyframe_normal[0] * Xcp[2] - keyframe_normal[2] * Xcp[0];
                (*jacobian)[2] = keyframe_normal[1] * Xcp[0] - keyframe_normal[0] * Xcp[1];

                if (translation_enabled)
                {
                  (*jacobian)[3] = keyframe_normal[0];
                  (*jacobian)[4] = keyframe_normal[1];
                  (*jacobian)[5] = keyframe_normal[2];
                }
              }
            }
          }
        }
      }
    }
  }
}

VULCAN_GLOBAL
void ComputeResidualsKernel(const Transform Tmc, const float* keyframe_depths,
    const Vector3f* keyframe_normals, const Projection keyframe_projection,
    int keyframe_width, int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const Projection frame_projection,
    int frame_width, int frame_height, float* residuals)
{
  float residual;
  const int frame_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int frame_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (frame_x < frame_width && frame_y < frame_height)
  {
    Evaluate<false>(frame_x, frame_y, Tmc, keyframe_depths, keyframe_normals,
        keyframe_projection, keyframe_width, keyframe_height, frame_depths,
        frame_normals, frame_projection, frame_width, frame_height, &residual,
        nullptr);

    residuals[frame_y * frame_width + frame_x] = residual;
  }
}

template <bool translation_enabled>
VULCAN_GLOBAL
void ComputeJacobianKernel(const Transform Tmc, const float* keyframe_depths,
    const Vector3f* keyframe_normals, const Projection keyframe_projection,
    int keyframe_width, int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const Projection frame_projection,
    int frame_width, int frame_height, Vector6f* jacobian)
{
  Vector6f result;
  const int frame_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int frame_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (frame_x < frame_width && frame_y < frame_height)
  {
    Evaluate<translation_enabled>(frame_x, frame_y, Tmc, keyframe_depths,
        keyframe_normals, keyframe_projection, keyframe_width, keyframe_height,
        frame_depths, frame_normals, frame_projection, frame_width,
        frame_height, nullptr, &result);

    // if (frame_x == 149 && frame_y == 0)
    // {
    //   printf("J: %f %f %f %f %f %f\n", result[0], result[1], result[2],
    //       result[3], result[4], result[5]);
    // }

    jacobian[frame_y * frame_width + frame_x] = result;
  }
}

template <bool translation_enabled>
VULCAN_GLOBAL
void ComputeSystemKernel(const Transform Tmc, const float* keyframe_depths,
    const Vector3f* keyframe_normals, const Projection keyframe_projection,
    int keyframe_width, int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const Projection frame_projection,
    int frame_width, int frame_height, float* hessian, float* gradient)
{
  VULCAN_SHARED float buffer1[256];
  VULCAN_SHARED float buffer2[256];
  VULCAN_SHARED float buffer3[256];

  float residual = 0;
  Vector6f jacobian = Vector6f::Zeros();

  const int frame_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int frame_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (frame_x < frame_width && frame_y < frame_height)
  {
    Evaluate<translation_enabled>(frame_x, frame_y, Tmc, keyframe_depths,
        keyframe_normals, keyframe_projection, keyframe_width, keyframe_height,
        frame_depths, frame_normals, frame_projection, frame_width,
        frame_height, &residual, &jacobian);
  }

  const int thread = threadIdx.y * blockDim.x + threadIdx.x;
  const int parameter_count = translation_enabled ? 6 : 3;

  for (int i = 0; i < parameter_count; i += 3)
  {
    buffer1[thread] = jacobian[i + 0] * residual;
    buffer2[thread] = jacobian[i + 1] * residual;
    buffer3[thread] = jacobian[i + 2] * residual;

    __syncthreads();

    if (thread < 128)
    {
      buffer1[thread] += buffer1[thread + 128];
      buffer2[thread] += buffer2[thread + 128];
      buffer3[thread] += buffer3[thread + 128];
    }

    __syncthreads();

    if (thread < 64)
    {
      buffer1[thread] += buffer1[thread + 64];
      buffer2[thread] += buffer2[thread + 64];
      buffer3[thread] += buffer3[thread + 64];
    }

    __syncthreads();

    if (thread < 32)
    {
      WarpReduce(buffer1, thread);
      WarpReduce(buffer2, thread);
      WarpReduce(buffer3, thread);
    }

    if (thread == 0)
    {
      atomicAdd(&gradient[i + 0], buffer1[thread]);
      atomicAdd(&gradient[i + 1], buffer2[thread]);
      atomicAdd(&gradient[i + 2], buffer3[thread]);
    }

    __syncthreads();
  }

  const int hessian_count = translation_enabled ? 21 : 6;
  float local_hessian[hessian_count];

  for (unsigned char r = 0, counter = 0; r < parameter_count; r++)
  {
    for (int c = 0; c <= r; c++, counter++)
    {
      local_hessian[counter] = jacobian[r] * jacobian[c];
    }
  }

  for (int i = 0; i < hessian_count; i += 3)
  {
    buffer1[thread] = local_hessian[i + 0];
    buffer2[thread] = local_hessian[i + 1];
    buffer3[thread] = local_hessian[i + 2];

    __syncthreads();

    if (thread < 128)
    {
      buffer1[thread] += buffer1[thread + 128];
      buffer2[thread] += buffer2[thread + 128];
      buffer3[thread] += buffer3[thread + 128];
    }

    __syncthreads();

    if (thread < 64)
    {
      buffer1[thread] += buffer1[thread + 64];
      buffer2[thread] += buffer2[thread + 64];
      buffer3[thread] += buffer3[thread + 64];
    }

    __syncthreads();

    if (thread < 32)
    {
      WarpReduce(buffer1, thread);
      WarpReduce(buffer2, thread);
      WarpReduce(buffer3, thread);
    }

    if (thread == 0)
    {
      atomicAdd(&hessian[i + 0], buffer1[thread]);
      atomicAdd(&hessian[i + 1], buffer2[thread]);
      atomicAdd(&hessian[i + 2], buffer3[thread]);
    }

    __syncthreads();
  }
}

} // namespace

void DepthTracker::ComputeResiduals(const Frame& frame,
    Buffer<float>& residuals) const
{
  residuals.Resize(GetResidualCount(frame));
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
  float* residuals_ptr = residuals.GetData();

  const dim3 threads(16, 16);
  const dim3 total(frame_width, frame_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(ComputeResidualsKernel, blocks, threads, 0, 0, Tmc,
      keyframe_depths, keyframe_normals, keyframe_projection, keyframe_width,
      keyframe_height, frame_depths, frame_normals, frame_projection,
      frame_width, frame_height, residuals_ptr);
}

void DepthTracker::ComputeJacobian(const Frame& frame,
    Buffer<Vector6f>& jacobian) const
{
  jacobian.Resize(GetResidualCount(frame));
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
  Vector6f* jacobian_ptr = jacobian.GetData();

  const dim3 threads(16, 16);
  const dim3 total(frame_width, frame_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  if (translation_enabled_)
  {
    CUDA_LAUNCH(ComputeJacobianKernel<true>, blocks, threads, 0, 0, Tmc,
        keyframe_depths, keyframe_normals, keyframe_projection, keyframe_width,
        keyframe_height, frame_depths, frame_normals, frame_projection,
        frame_width, frame_height, jacobian_ptr);
  }
  else
  {
    CUDA_LAUNCH(ComputeJacobianKernel<false>, blocks, threads, 0, 0, Tmc,
        keyframe_depths, keyframe_normals, keyframe_projection, keyframe_width,
        keyframe_height, frame_depths, frame_normals, frame_projection,
        frame_width, frame_height, jacobian_ptr);
  }
}

void DepthTracker::ComputeSystem(const Frame& frame)
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
  float* hessian = hessian_.GetData();
  float* gradient = gradient_.GetData();

  thrust::device_ptr<float> dh(hessian);
  thrust::device_ptr<float> dg(gradient);
  thrust::fill(dh, dh + hessian_.GetSize(), 0.0f);
  thrust::fill(dg, dg + gradient_.GetSize(), 0.0f);

  const dim3 threads(16, 16);
  const dim3 total(frame_width, frame_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  if (translation_enabled_)
  {
    CUDA_LAUNCH(ComputeSystemKernel<true>, blocks, threads, 0, 0, Tmc,
        keyframe_depths, keyframe_normals, keyframe_projection, keyframe_width,
        keyframe_height, frame_depths, frame_normals, frame_projection,
        frame_width, frame_height, hessian, gradient);
  }
  else
  {
    CUDA_LAUNCH(ComputeSystemKernel<false>, blocks, threads, 0, 0, Tmc,
        keyframe_depths, keyframe_normals, keyframe_projection, keyframe_width,
        keyframe_height, frame_depths, frame_normals, frame_projection,
        frame_width, frame_height, hessian, gradient);
  }
}

} // namespace vulcan