#include <vulcan/depth_tracker.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <vulcan/device.h>
#include <vulcan/frame.h>
#include <vulcan/projection.h>
#include <vulcan/transform.h>

namespace vulcan
{

VULCAN_DEVICE
inline void WarpReduce(volatile float* buffer, int thread)
{
  buffer[thread] += buffer[thread + 32];
  buffer[thread] += buffer[thread + 16];
  buffer[thread] += buffer[thread +  8];
  buffer[thread] += buffer[thread +  4];
  buffer[thread] += buffer[thread +  2];
  buffer[thread] += buffer[thread +  1];
}

namespace
{

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

  const int frame_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int frame_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int thread = threadIdx.y * blockDim.x + threadIdx.x;

  float residual = 0;
  Vector6f dfdx = Vector6f::Zeros();

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
            const Vector3f Ymp = keyframe_projection.Unproject(keyframe_uv, keyframe_depth);
            const Vector3f delta = Xmp - Ymp;

            if (delta.SquaredNorm() < 0.05)
            {
              residual = delta.Dot(keyframe_normal);

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
  }

  const int parameter_count = translation_enabled ? 6 : 3;

  for (int i = 0; i < parameter_count; i += 3)
  {
    buffer1[thread] = dfdx[i + 0] * residual;
    buffer2[thread] = dfdx[i + 1] * residual;
    buffer3[thread] = dfdx[i + 2] * residual;

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
      local_hessian[counter] = dfdx[r] * dfdx[c];
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