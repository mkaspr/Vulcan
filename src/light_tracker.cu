#include <vulcan/light_tracker.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <vulcan/device.h>
#include <vulcan/frame.h>
#include <vulcan/projection.h>
#include <vulcan/transform.h>
#include <vulcan/util.cuh>

#include <opencv2/opencv.hpp>

namespace vulcan
{

namespace
{


VULCAN_DEVICE
float Sample(int w, int h, const float* values, float u, float v)
{
  const int x = floorf(u - 0.5f);
  const int y = floorf(v - 0.5f);

  VULCAN_DEBUG(x >= 0 && x < w - 1 && y >= 0 && y < h - 1);

  const float v00 = values[(y + 0) * w + (x + 0)];
  const float v01 = values[(y + 0) * w + (x + 1)];
  const float v10 = values[(y + 1) * w + (x + 0)];
  const float v11 = values[(y + 1) * w + (x + 1)];

  const float u1 = u - (x + 0.5f);
  const float v1 = v - (y + 0.5f);
  const float u0 = 1.0f - u1;
  const float v0 = 1.0f - v1;

  const float w00 = v0 * u0;
  const float w01 = v0 * u1;
  const float w10 = v1 * u0;
  const float w11 = v1 * u1;

  return w00 * v00 +  w01 * v01 +  w10 * v10 +  w11 * v11;
}

template <bool translation_enabled>
VULCAN_DEVICE
void Evaluate(int keyframe_x, int keyframe_y, const Transform& Tcm,
    const float* keyframe_depths, const Vector3f* keyframe_normals,
    const float* keyframe_albedos, const Projection& keyframe_projection,
    int keyframe_width, int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const float* frame_intensities,
    const float* frame_gradient_x, const float* frame_gradient_y,
    const Projection& frame_projection, int frame_width, int frame_height,
    const Light& light, float* residual, Vector6f* jacobian)
{
  VULCAN_DEBUG(keyframe_x >= 0 && keyframe_x < keyframe_width);
  VULCAN_DEBUG(keyframe_y >= 0 && keyframe_y < keyframe_height);

  const int keyframe_index = keyframe_y * keyframe_width + keyframe_x;
  const float keyframe_depth = keyframe_depths[keyframe_index];

  if (residual) *residual = 0;
  if (jacobian) *jacobian = Vector6f::Zeros();

  if (keyframe_depth > 0)
  {
    const float keyframe_u = keyframe_x + 0.5f;
    const float keyframe_v = keyframe_y + 0.5f;
    const Vector3f Xmp = keyframe_projection.Unproject(keyframe_u, keyframe_v, keyframe_depth);
    const Vector3f Xcp = Vector3f(Tcm * Vector4f(Xmp, 1));
    const Vector2f frame_uv = frame_projection.Project(Xcp);

    if (frame_uv[0] >= 0.5f && frame_uv[0] < frame_width  - 0.5f &&
        frame_uv[1] >= 0.5f && frame_uv[1] < frame_height - 0.5f)
    {
      const int frame_x = frame_uv[0];
      const int frame_y = frame_uv[1];
      const int frame_index = frame_y * frame_width + frame_x;
      const float frame_depth = frame_depths[frame_index];

      if (fabsf(frame_depth - Xcp[2]) < 0.1f)
      {
        const Vector3f frame_normal = frame_normals[frame_index];
        Vector3f keyframe_normal = keyframe_normals[keyframe_index];
        keyframe_normal = Vector3f(Tcm * Vector4f(keyframe_normal, 0));

        if (keyframe_normal.SquaredNorm() > 0.5f &&
            frame_normal.Dot(keyframe_normal) > 0.5f)
        {
          const float aa = keyframe_albedos[keyframe_index];

          if (aa > 0)
          {
            const float shading = light.GetShading(Xcp, keyframe_normal);
            const float Im = shading * aa;

            const float Ic = Sample(frame_width, frame_height,
                frame_intensities, frame_uv[0], frame_uv[1]);

            if (residual) *residual = Ic - Im;

            if (jacobian)
            {
              Vector6f dfdx = Vector6f::Zeros();

              const float px = Xcp[0];
              const float py = Xcp[1];
              const float pz = Xcp[2];

              const float nx = keyframe_normal[0];
              const float ny = keyframe_normal[1];
              const float nz = keyframe_normal[2];

              const float lx = light.GetPosition()[0];
              const float ly = light.GetPosition()[1];
              const float lz = light.GetPosition()[2];

              const float ii = light.GetIntensity();

              const float fx = frame_projection.GetFocalLength()[0];
              const float fy = frame_projection.GetFocalLength()[1];
              const float cx = frame_projection.GetCenterPoint()[0];
              const float cy = frame_projection.GetCenterPoint()[1];

              const float gx = Sample(frame_width, frame_height, frame_gradient_x,
                  frame_uv[0], frame_uv[1]);

              const float gy = Sample(frame_width, frame_height, frame_gradient_y,
                  frame_uv[0], frame_uv[1]);

              dfdx[0] = gy*((cy*py-fy*pz)/pz-py*1.0f/(pz*pz)*(cy*pz+fy*py))+gx*((cx*py)/pz-py*1.0f/(pz*pz)*(cx*pz+fx*px))+(aa*ii*(nz*(ly-py)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))-ny*(lz-pz)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))-ny*pz*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nz*py*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nx*(pz*(ly-py)*2.0f-py*(lz-pz)*2.0f)*(lx-px)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)+ny*(pz*(ly-py)*2.0f-py*(lz-pz)*2.0f)*(ly-py)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)+nz*(pz*(ly-py)*2.0f-py*(lz-pz)*2.0f)*(lz-pz)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)))/(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+aa*ii*(pz*(ly-py)*2.0f-py*(lz-pz)*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),2.0f)*(nx*(lx-px)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+ny*(ly-py)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nz*(lz-pz)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f)));
              dfdx[1] = -gx*((cx*px-fx*pz)/pz-px*1.0f/(pz*pz)*(cx*pz+fx*px))-gy*((cy*px)/pz-px*1.0f/(pz*pz)*(cy*pz+fy*py))-(aa*ii*(nz*(lx-px)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))-nx*(lz-pz)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))-nx*pz*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nz*px*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nx*(pz*(lx-px)*2.0f-px*(lz-pz)*2.0f)*(lx-px)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)+ny*(pz*(lx-px)*2.0f-px*(lz-pz)*2.0f)*(ly-py)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)+nz*(pz*(lx-px)*2.0f-px*(lz-pz)*2.0f)*(lz-pz)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)))/(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))-aa*ii*(pz*(lx-px)*2.0f-px*(lz-pz)*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),2.0f)*(nx*(lx-px)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+ny*(ly-py)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nz*(lz-pz)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f)));
              dfdx[2] = (aa*ii*(ny*(lx-px)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))-nx*(ly-py)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))-nx*py*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+ny*px*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nx*(py*(lx-px)*2.0f-px*(ly-py)*2.0f)*(lx-px)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)+ny*(py*(lx-px)*2.0f-px*(ly-py)*2.0f)*(ly-py)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)+nz*(py*(lx-px)*2.0f-px*(ly-py)*2.0f)*(lz-pz)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)))/(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))-(fx*gx*py)/pz+(fy*gy*px)/pz+aa*ii*(py*(lx-px)*2.0f-px*(ly-py)*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),2.0f)*(nx*(lx-px)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+ny*(ly-py)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nz*(lz-pz)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f)));

              if (translation_enabled)
              {
                dfdx[3] = (fx*gx)/pz-(aa*ii*(-nx*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nx*(lx-px)*(lx*2.0f-px*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)+ny*(ly-py)*(lx*2.0f-px*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)+nz*(lz-pz)*(lx*2.0f-px*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)))/(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))-aa*ii*(lx*2.0f-px*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),2.0f)*(nx*(lx-px)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+ny*(ly-py)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nz*(lz-pz)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f)));
                dfdx[4] = (fy*gy)/pz-(aa*ii*(-ny*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nx*(lx-px)*(ly*2.0f-py*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)+ny*(ly-py)*(ly*2.0f-py*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)+nz*(lz-pz)*(ly*2.0f-py*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)))/(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))-aa*ii*(ly*2.0f-py*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),2.0f)*(nx*(lx-px)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+ny*(ly-py)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nz*(lz-pz)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f)));
                dfdx[5] = gx*(cx/pz-1.0f/(pz*pz)*(cx*pz+fx*px))+gy*(cy/pz-1.0f/(pz*pz)*(cy*pz+fy*py))-(aa*ii*(-nz*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nx*(lx-px)*(lz*2.0f-pz*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)+ny*(ly-py)*(lz*2.0f-pz*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)+nz*(lz-pz)*(lz*2.0f-pz*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),3.0f/2.0f)*(1.0f/2.0f)))/(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))-aa*ii*(lz*2.0f-pz*2.0f)*1.0f/powf(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f),2.0f)*(nx*(lx-px)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+ny*(ly-py)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f))+nz*(lz-pz)*1.0f/sqrt(powf(lx-px,2.0f)+powf(ly-py,2.0f)+powf(lz-pz,2.0f)));
              }

              *jacobian = dfdx;
            }
          }
        }
      }
    }
  }
}

VULCAN_GLOBAL
void ComputeResidualsKernel(const Transform Tcm, const float* keyframe_depths,
    const Vector3f* keyframe_normals, const float* keyframe_albedos,
    const Projection keyframe_projection, int keyframe_width,
    int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const float* frame_intensities,
    const Projection frame_projection, int frame_width, int frame_height,
    const Light light, float* residuals)
{
  const int keyframe_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int keyframe_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (keyframe_x < keyframe_width && keyframe_y < keyframe_height)
  {
    float residual = 0;

    Evaluate<false>(keyframe_x, keyframe_y, Tcm, keyframe_depths,
        keyframe_normals, keyframe_albedos, keyframe_projection, keyframe_width,
        keyframe_height, frame_depths, frame_normals, frame_intensities,
        nullptr, nullptr, frame_projection, frame_width, frame_height, light,
        &residual, nullptr);

    residuals[keyframe_y * keyframe_width + keyframe_x] = residual;
  }
}

template <bool translation_enabled>
VULCAN_GLOBAL
void ComputeJacobianKernel(const Transform Tcm, const float* keyframe_depths,
    const Vector3f* keyframe_normals, const float* keyframe_albedos,
    const Projection keyframe_projection, int keyframe_width,
    int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const float* frame_intensities,
    const float* frame_gradient_x, const float* frame_gradient_y,
    const Projection frame_projection, int frame_width, int frame_height,
    const Light light, Vector6f* jacobian)
{
  const int keyframe_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int keyframe_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (keyframe_x < keyframe_width && keyframe_y < keyframe_height)
  {
    Vector6f dfdx = Vector6f::Zeros();

    Evaluate<translation_enabled>(keyframe_x, keyframe_y, Tcm, keyframe_depths,
        keyframe_normals, keyframe_albedos, keyframe_projection, keyframe_width,
        keyframe_height, frame_depths, frame_normals, frame_intensities,
        frame_gradient_x, frame_gradient_y, frame_projection, frame_width,
        frame_height, light, nullptr, &dfdx);

    jacobian[keyframe_y * keyframe_width + keyframe_x] = dfdx;
  }
}

template <bool translation_enabled>
VULCAN_GLOBAL
void ComputeSystemKernel(const Transform Tcm, const float* keyframe_depths,
    const Vector3f* keyframe_normals, const float* keyframe_albedos,
    const Projection keyframe_projection, int keyframe_width,
    int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const float* frame_intensities,
    const float* frame_gradient_x, const float* frame_gradient_y,
    const Projection frame_projection, int frame_width, int frame_height,
    const Light light, float* hessian, float* gradient)
{
  VULCAN_SHARED float buffer1[256];
  VULCAN_SHARED float buffer2[256];
  VULCAN_SHARED float buffer3[256];

  const int keyframe_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int keyframe_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int thread = threadIdx.y * blockDim.x + threadIdx.x;

  float residual = 0;
  Vector6f dfdx = Vector6f::Zeros();

  if (keyframe_x < keyframe_width && keyframe_y < keyframe_height)
  {
    Evaluate<translation_enabled>(keyframe_x, keyframe_y, Tcm, keyframe_depths,
    keyframe_normals, keyframe_albedos, keyframe_projection, keyframe_width,
    keyframe_height, frame_depths, frame_normals, frame_intensities,
    frame_gradient_x, frame_gradient_y, frame_projection, frame_width,
    frame_height, light, &residual, &dfdx);
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

void LightTracker::ComputeKeyframeIntensities()
{
  keyframe_->color_image->ConvertTo(keyframe_intensities_);
}

void LightTracker::ComputeFrameIntensities(const Frame& frame)
{
  frame.color_image->ConvertTo(frame_intensities_);
}

void LightTracker::ComputeFrameGradients(const Frame& frame)
{
  frame_intensities_.GetGradients(frame_gradient_x_, frame_gradient_y_);
}

void LightTracker::ComputeResiduals(const Frame& frame,
    Buffer<float>& residuals)
{
  ComputeKeyframeIntensities();
  ComputeFrameIntensities(frame);

  const int frame_width = frame.depth_image->GetWidth();
  const int frame_height = frame.depth_image->GetHeight();
  const int keyframe_width = keyframe_->depth_image->GetWidth();
  const int keyframe_height = keyframe_->depth_image->GetHeight();
  const float* frame_depths = frame.depth_image->GetData();
  const float* keyframe_intensities = keyframe_intensities_.GetData();
  const float* frame_intensities = frame_intensities_.GetData();
  const float* keyframe_depths = keyframe_->depth_image->GetData();
  const Vector3f* keyframe_normals = keyframe_->normal_image->GetData();
  const Vector3f* frame_normals = frame.normal_image->GetData();
  const Projection& frame_projection = frame.projection;
  const Projection& keyframe_projection = keyframe_->projection;
  const Transform Tcm = frame.Twc.Inverse() * keyframe_->Twc;

  residuals.Resize(keyframe_->depth_image->GetTotal());
  float* d_residuals = residuals.GetData();

  const dim3 threads(16, 16);
  const dim3 total(keyframe_width, keyframe_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(ComputeResidualsKernel, blocks, threads, 0, 0, Tcm,
      keyframe_depths, keyframe_normals, keyframe_intensities,
      keyframe_projection, keyframe_width, keyframe_height, frame_depths,
      frame_normals, frame_intensities, frame_projection, frame_width,
      frame_height, light_, d_residuals);
}

void LightTracker::ComputeJacobian(const Frame& frame,
    Buffer<Vector6f>& jacobian)
{
  ComputeKeyframeIntensities();
  ComputeFrameIntensities(frame);
  ComputeFrameGradients(frame);

  const int frame_width = frame.depth_image->GetWidth();
  const int frame_height = frame.depth_image->GetHeight();
  const int keyframe_width = keyframe_->depth_image->GetWidth();
  const int keyframe_height = keyframe_->depth_image->GetHeight();
  const float* frame_depths = frame.depth_image->GetData();
  const float* keyframe_intensities = keyframe_intensities_.GetData();
  const float* frame_intensities = frame_intensities_.GetData();
  const float* keyframe_depths = keyframe_->depth_image->GetData();
  const float* frame_gradient_x = frame_gradient_x_.GetData();
  const float* frame_gradient_y = frame_gradient_y_.GetData();
  const Vector3f* keyframe_normals = keyframe_->normal_image->GetData();
  const Vector3f* frame_normals = frame.normal_image->GetData();
  const Projection& frame_projection = frame.projection;
  const Projection& keyframe_projection = keyframe_->projection;
  const Transform Tcm = frame.Twc.Inverse() * keyframe_->Twc;

  jacobian.Resize(keyframe_->depth_image->GetTotal());
  Vector6f* d_jacobian = jacobian.GetData();

  const dim3 threads(16, 16);
  const dim3 total(keyframe_width, keyframe_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  if (translation_enabled_)
  {
    CUDA_LAUNCH(ComputeJacobianKernel<true>, blocks, threads, 0, 0, Tcm,
        keyframe_depths, keyframe_normals, keyframe_intensities,
        keyframe_projection, keyframe_width, keyframe_height, frame_depths,
        frame_normals, frame_intensities, frame_gradient_x, frame_gradient_y,
        frame_projection, frame_width, frame_height, light_, d_jacobian);
  }
  else
  {
    CUDA_LAUNCH(ComputeJacobianKernel<false>, blocks, threads, 0, 0, Tcm,
        keyframe_depths, keyframe_normals, keyframe_intensities,
        keyframe_projection, keyframe_width, keyframe_height, frame_depths,
        frame_normals, frame_intensities, frame_gradient_x, frame_gradient_y,
        frame_projection, frame_width, frame_height, light_, d_jacobian);
  }
}

void LightTracker::ComputeSystem(const Frame& frame)
{
  const int frame_width = frame.depth_image->GetWidth();
  const int frame_height = frame.depth_image->GetHeight();
  const int keyframe_width = keyframe_->depth_image->GetWidth();
  const int keyframe_height = keyframe_->depth_image->GetHeight();
  const float* frame_depths = frame.depth_image->GetData();
  const float* keyframe_intensities = keyframe_intensities_.GetData();
  const float* frame_intensities = frame_intensities_.GetData();
  const float* keyframe_depths = keyframe_->depth_image->GetData();
  const float* frame_gradient_x = frame_gradient_x_.GetData();
  const float* frame_gradient_y = frame_gradient_y_.GetData();
  const Vector3f* keyframe_normals = keyframe_->normal_image->GetData();
  const Vector3f* frame_normals = frame.normal_image->GetData();
  const Projection& frame_projection = frame.projection;
  const Projection& keyframe_projection = keyframe_->projection;
  const Transform Tcm = frame.Twc.Inverse() * keyframe_->Twc;
  float* hessian = hessian_.GetData();
  float* gradient = gradient_.GetData();

  thrust::device_ptr<float> dh(hessian);
  thrust::device_ptr<float> dg(gradient);
  thrust::fill(dh, dh + hessian_.GetSize(), 0.0f);
  thrust::fill(dg, dg + gradient_.GetSize(), 0.0f);

  const dim3 threads(16, 16);
  const dim3 total(keyframe_width, keyframe_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  if (translation_enabled_)
  {
    CUDA_LAUNCH(ComputeSystemKernel<true>, blocks, threads, 0, 0, Tcm,
        keyframe_depths, keyframe_normals, keyframe_intensities,
        keyframe_projection, keyframe_width, keyframe_height, frame_depths,
        frame_normals, frame_intensities, frame_gradient_x, frame_gradient_y,
        frame_projection, frame_width, frame_height, light_, hessian, gradient);
  }
  else
  {
    CUDA_LAUNCH(ComputeSystemKernel<false>, blocks, threads, 0, 0, Tcm,
        keyframe_depths, keyframe_normals, keyframe_intensities,
        keyframe_projection, keyframe_width, keyframe_height, frame_depths,
        frame_normals, frame_intensities, frame_gradient_x, frame_gradient_y,
        frame_projection, frame_width, frame_height, light_, hessian, gradient);
  }
}

} // namespace vulcan