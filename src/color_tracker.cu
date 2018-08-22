#include <vulcan/color_tracker.h>
#include <vulcan/device.h>
#include <vulcan/frame.h>
#include <vulcan/projection.h>
#include <vulcan/transform.h>

namespace vulcan
{
namespace
{

VULCAN_GLOBAL
void ComputeIntensitiesKernel(int total, const Vector3f* colors,
    float* intensities)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < total)
  {
    const Vector3f color = colors[index];
    intensities[index] = (color[0] + color[1] + color[2]) / 3.0f;
  }
}

template <int BLOCK_DIM>
VULCAN_GLOBAL
void ComputeGradientsKernel(int width, int height, const float* intensities,
    float* gradient_x, float* gradient_y)
{
  const int buffer_dim = BLOCK_DIM + 2;
  const int buffer_size = buffer_dim * buffer_dim;
  VULCAN_SHARED float buffer[buffer_size];
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int block_size = blockDim.x * blockDim.y;
  int sindex = threadIdx.y * blockDim.x + threadIdx.x;

  do
  {
    float intensity = 0;
    const int gx = (blockIdx.x * blockDim.x - 1) + (sindex % buffer_dim);
    const int gy = (blockIdx.y * blockDim.y - 1) + (sindex / buffer_dim);

    if (gx >= 0 && gx < width && gy >= 0 && gy < height)
    {
      intensity = intensities[gy * width + gx];
    }

    buffer[sindex] = intensity;
    sindex += block_size;
  }
  while (sindex < buffer_size);

  __syncthreads();

  if (x < width && y < height)
  {
    float gx = 0;
    float gy = 0;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
      const int tx = threadIdx.x;
      const int ty = threadIdx.y;

      const float i00 = 0.25f * buffer[(ty + 0) * buffer_dim + (tx + 0)];
      const float i01 = 0.50f * buffer[(ty + 0) * buffer_dim + (tx + 1)];
      const float i02 = 0.25f * buffer[(ty + 0) * buffer_dim + (tx + 2)];

      const float i10 = 0.50f * buffer[(ty + 1) * buffer_dim + (tx + 0)];
      const float i12 = 0.50f * buffer[(ty + 1) * buffer_dim + (tx + 2)];

      const float i20 = 0.25f * buffer[(ty + 2) * buffer_dim + (tx + 0)];
      const float i21 = 0.50f * buffer[(ty + 2) * buffer_dim + (tx + 1)];
      const float i22 = 0.25f * buffer[(ty + 2) * buffer_dim + (tx + 2)];

      gx = (i00 + i10 + i20) - (i02 + i12 + i22);
      gy = (i00 + i01 + i02) - (i20 + i21 + i22);
    }

    const int index = y * width + x;
    gradient_x[index] = gx;
    gradient_y[index] = gy;
  }
}

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

  const float u1 = u - x;
  const float u0 = 1 - u1;
  const float v1 = v - y;
  const float v0 = 1 - v1;

  const float w00 = v0 * u0;
  const float w01 = v0 * u1;
  const float w10 = v1 * u0;
  const float w11 = v1 * u1;

  return w00 * v00 +  w01 * v01 +  w10 * v10 +  w11 * v11;
}

VULCAN_GLOBAL
void ComputeResidualKernel(const Transform Tcm, const float* keyframe_depths,
    const Vector3f* keyframe_normals, const float* keyframe_intensities,
    const Projection keyframe_projection, int keyframe_width,
    int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const float* frame_intensities,
    const Projection frame_projection, int frame_width, int frame_height,
    double* residuals)
{
  const int keyframe_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int keyframe_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (keyframe_x < keyframe_width && keyframe_y < keyframe_height)
  {
    double residual = 0;
    const int keyframe_index = keyframe_y * keyframe_width + keyframe_x;
    const float keyframe_depth = keyframe_depths[keyframe_index];

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

        if (fabsf(frame_depth - Xcp[2]) < 0.01)
        {
          const Vector3f frame_normal = frame_normals[frame_index];
          const Vector3f keyframe_normal = keyframe_normals[keyframe_index];

          if (keyframe_normal.SquaredNorm() > 0 &&
              frame_normal.Dot(keyframe_normal) > 0.5f)
          {
            const float Im = keyframe_intensities[keyframe_index];

            const float Ic = Sample(frame_width, frame_height,
                frame_intensities, frame_uv[0], frame_uv[1]);

            residual = Ic - Im;
          }
        }
      }
    }

    residuals[keyframe_index] = residual;
  }
}

template <bool translation_enabled>
VULCAN_GLOBAL
void ComputeJacobianKernel(const Transform Tcm, const float* keyframe_depths,
    const Vector3f* keyframe_normals, const Projection keyframe_projection,
    int keyframe_width, int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const float* frame_gradient_x,
    const float* frame_gradient_y, const Projection frame_projection,
    int frame_width, int frame_height, double* jacobian)
{
  const int keyframe_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int keyframe_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (keyframe_x < keyframe_width && keyframe_y < keyframe_height)
  {
    Vector6f dfdx = Vector6f::Zeros();
    const int keyframe_index = keyframe_y * keyframe_width + keyframe_x;
    const float keyframe_depth = keyframe_depths[keyframe_index];

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

        if (fabsf(frame_depth - Xcp[2]) < 0.01)
        {
          const Vector3f frame_normal = frame_normals[frame_index];
          const Vector3f keyframe_normal = keyframe_normals[keyframe_index];

          if (keyframe_normal.SquaredNorm() > 0 &&
              frame_normal.Dot(keyframe_normal) > 0.5f)
          {
            const float x = Xmp[0];
            const float y = Xmp[1];
            const float z = Xmp[2];

            const float fx = frame_projection.GetFocalLength()[0];
            const float fy = frame_projection.GetFocalLength()[1];

            const float gx = Sample(frame_width, frame_height, frame_gradient_x,
                frame_uv[0], frame_uv[1]);

            const float gy = Sample(frame_width, frame_height, frame_gradient_y,
                frame_uv[0], frame_uv[1]);

            dfdx[0] = -fy*gy-y*(fx*gx*x*1.0/(z*z)+fy*gy*y*1.0/(z*z));
            dfdx[1] = fx*gx+x*(fx*gx*x*1.0/(z*z)+fy*gy*y*1.0/(z*z));
            dfdx[2] = (fy*gy*x)/z-(fx*gx*y)/z;

            if (translation_enabled)
            {
              dfdx[3] = (fx*gx)/z;
              dfdx[4] = (fy*gy)/z;
              dfdx[5] = -fx*gx*x*1.0/(z*z)-fy*gy*y*1.0/(z*z);
            }
          }
        }
      }
    }

    const int residual_count = keyframe_width * keyframe_height;
    jacobian[0 * residual_count + keyframe_index] = dfdx[0];
    jacobian[1 * residual_count + keyframe_index] = dfdx[1];
    jacobian[2 * residual_count + keyframe_index] = dfdx[2];

    if (translation_enabled)
    {
      jacobian[3 * residual_count + keyframe_index] = dfdx[3];
      jacobian[4 * residual_count + keyframe_index] = dfdx[4];
      jacobian[5 * residual_count + keyframe_index] = dfdx[5];
    }
  }
}

} // namespace

void ColorTracker::ComputeKeyframeIntensities()
{
  const int width = keyframe_->depth_image->GetWidth();
  const int height = keyframe_->depth_image->GetHeight();
  keyframe_intensities_.Resize(width, height);

  const Vector3f* colors = keyframe_->color_image->GetData();
  float* intensities = keyframe_intensities_.GetData();

  const int threads = 512;
  const int total = width * height;
  const int blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(ComputeIntensitiesKernel, blocks, threads, 0, 0, total, colors,
      intensities);
}

void ColorTracker::ComputeFrameIntensities(const Frame& frame)
{
  const int width = frame.depth_image->GetWidth();
  const int height = frame.depth_image->GetHeight();
  frame_intensities_.Resize(width, height);

  const Vector3f* colors = frame.color_image->GetData();
  float* intensities = frame_intensities_.GetData();

  const int threads = 512;
  const int total = width * height;
  const int blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(ComputeIntensitiesKernel, blocks, threads, 0, 0, total, colors,
      intensities);
}

void ColorTracker::ComputeFrameGradients(const Frame& frame)
{
  const int width = frame.depth_image->GetWidth();
  const int height = frame.depth_image->GetHeight();
  frame_gradient_x_.Resize(width, height);
  frame_gradient_y_.Resize(width, height);

  const float* intensities = frame_intensities_.GetData();
  float* gradient_x = frame_gradient_x_.GetData();
  float* gradient_y = frame_gradient_y_.GetData();

  const dim3 threads(16, 16);
  const dim3 total(width, height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(ComputeGradientsKernel<16>, blocks, threads, 0, 0, width, height,
      intensities, gradient_x, gradient_y);
}

void ColorTracker::ComputeResidual(const Frame& frame)
{
  const int frame_width = frame.depth_image->GetWidth();
  const int frame_height = frame.depth_image->GetHeight();
  const int keyframe_width = keyframe_->depth_image->GetWidth();
  const int keyframe_height = keyframe_->depth_image->GetHeight();
  const float* frame_depths = frame.depth_image->GetData();
  const float* keyframe_depths = keyframe_->depth_image->GetData();
  const float* keyframe_intensities = keyframe_intensities_.GetData();
  const float* frame_intensities = frame_intensities_.GetData();
  const Vector3f* keyframe_normals = keyframe_->normal_image->GetData();
  const Vector3f* frame_normals = frame.normal_image->GetData();
  const Projection& frame_projection = frame.projection;
  const Projection& keyframe_projection = keyframe_->projection;
  const Transform Tcm = frame.Tcw * keyframe_->Tcw.Inverse();
  double* residuals = residuals_.GetData();

  const dim3 threads(16, 16);
  const dim3 total(keyframe_width, keyframe_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(ComputeResidualKernel, blocks, threads, 0, 0, Tcm,
      keyframe_depths, keyframe_normals, keyframe_intensities,
      keyframe_projection, keyframe_width, keyframe_height, frame_depths,
      frame_normals, frame_intensities, frame_projection, frame_width,
      frame_height, residuals);
}

void ColorTracker::ComputeJacobian(const Frame& frame)
{
  const int frame_width = frame.depth_image->GetWidth();
  const int frame_height = frame.depth_image->GetHeight();
  const int keyframe_width = keyframe_->depth_image->GetWidth();
  const int keyframe_height = keyframe_->depth_image->GetHeight();
  const float* frame_depths = frame.depth_image->GetData();
  const float* keyframe_depths = keyframe_->depth_image->GetData();
  const float* frame_gradient_x = frame_gradient_x_.GetData();
  const float* frame_gradient_y = frame_gradient_y_.GetData();
  const Vector3f* keyframe_normals = keyframe_->normal_image->GetData();
  const Vector3f* frame_normals = frame.normal_image->GetData();
  const Projection& frame_projection = frame.projection;
  const Projection& keyframe_projection = keyframe_->projection;
  const Transform Tcm = frame.Tcw * keyframe_->Tcw.Inverse();
  double* jacobian = jacobian_.GetData();

  const dim3 threads(16, 16);
  const dim3 total(keyframe_width, keyframe_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  if (translation_enabled_)
  {
    CUDA_LAUNCH(ComputeJacobianKernel<true>, blocks, threads, 0, 0, Tcm,
        keyframe_depths, keyframe_normals, keyframe_projection, keyframe_width,
        keyframe_height, frame_depths, frame_normals, frame_gradient_x,
        frame_gradient_y, frame_projection, frame_width, frame_height,
        jacobian);
  }
  else
  {
    CUDA_LAUNCH(ComputeJacobianKernel<false>, blocks, threads, 0, 0, Tcm,
        keyframe_depths, keyframe_normals, keyframe_projection, keyframe_width,
        keyframe_height, frame_depths, frame_normals, frame_gradient_x,
        frame_gradient_y, frame_projection, frame_width, frame_height,
        jacobian);
  }
}

} // namespace vulcan