#include <vulcan/frame.cuh>
#include <vulcan/projection.h>

namespace vulcan
{

template <int PATCH_SIZE>
VULCAN_GLOBAL
void ComputeNormalsKernel(const float* depths, const Projection projection,
    Vector3f* normals, int image_width, int image_height)
{
  const int pad = 2;
  const int resolution = (PATCH_SIZE + 2 * pad);
  const int shared_size = resolution * resolution;
  VULCAN_SHARED float shared[shared_size];

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  int shared_index = threadIdx.y * blockDim.x + threadIdx.x;

  while (shared_index < shared_size)
  {
    float depth = 0;
    const int bx = blockIdx.x * blockDim.x - pad + (shared_index % resolution);
    const int by = blockIdx.y * blockDim.y - pad + (shared_index / resolution);

    if (bx >= 0 && bx < image_width && by >= 0 && by < image_height)
    {
      depth = depths[by * image_width + bx];
    }

    shared[shared_index] = depth;
    shared_index += blockDim.x * blockDim.y;
  }

  __syncthreads();

  const float depth = shared[(threadIdx.y + pad) * resolution + (threadIdx.x + pad)];

  if (x < image_width && y < image_height)
  {
    Vector3f normal(0, 0, 0);

    if (depth > 0)
    {
      float d;
      Vector2f uv;

      uv[0] = (x + 0) + 0.5f;
      uv[1] = (y + 0) + 0.5f;
      const Vector3f z0 = projection.Unproject(uv, depth);

      Vector3f x0;
      d = shared[(threadIdx.y + pad) * resolution + (threadIdx.x + 0)];

      if (d == 0)
      {
        x0 = z0;
      }
      else
      {
        uv[0] = (x - pad) + 0.5f;
        uv[1] = (y + 0) + 0.5f;
        x0 = projection.Unproject(uv) * d;
      }

      Vector3f x1;
      d = shared[(threadIdx.y + pad) * resolution + (threadIdx.x + 2 * pad)];

      if (d == 0)
      {
        x1 = z0;
      }
      else
      {
        uv[0] = (x + pad) + 0.5f;
        uv[1] = (y + 0) + 0.5f;
        x1 = projection.Unproject(uv) * d;
      }

      Vector3f y0;
      d = shared[(threadIdx.y + 0) * resolution + (threadIdx.x + pad)];

      if (d == 0)
      {
        y0 = z0;
      }
      else
      {
        uv[0] = (x + 0) + 0.5f;
        uv[1] = (y - pad) + 0.5f;
        y0 = projection.Unproject(uv) * d;
      }

      Vector3f y1;
      d = shared[(threadIdx.y + 2 * pad) * resolution + (threadIdx.x + pad)];

      if (d == 0)
      {
        y1 = z0;
      }
      else
      {
        uv[0] = (x + 0) + 0.5f;
        uv[1] = (y + pad) + 0.5f;
        y1 = projection.Unproject(uv) * d;
      }

      const Vector3f dx = x0 - x1;
      const Vector3f dy = y0 - y1;

      if (dx.SquaredNorm() > 0 && dy.SquaredNorm() > 0)
      {
        normal = dy.Cross(dx);
        normal.Normalize();
      }
    }

    const int output = y * image_width + x;
    normals[output] = normal;
  }
}

template <int PATCH_SIZE>
VULCAN_GLOBAL
void FilterDepthsKernel(int image_width, int image_height, const float* src,
    float* dst)
{
  const int pad = 3;
  const int resolution = (PATCH_SIZE + 2 * pad);
  const int shared_size = resolution * resolution;
  VULCAN_SHARED float shared[shared_size];

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  int shared_index = threadIdx.y * blockDim.x + threadIdx.x;

  while (shared_index < shared_size)
  {
    float depth = 0;
    const int bx = blockIdx.x * blockDim.x - pad + (shared_index % resolution);
    const int by = blockIdx.y * blockDim.y - pad + (shared_index / resolution);

    if (bx >= 0 && bx < image_width && by >= 0 && by < image_height)
    {
      depth = src[by * image_width + bx];
    }

    shared[shared_index] = depth;
    shared_index += blockDim.x * blockDim.y;
  }

  __syncthreads();

  if (x < image_width && y < image_height)
  {
    const int tx = threadIdx.x + pad;
    const int ty = threadIdx.y + pad;
    const float d0 = shared[ty * resolution + tx];
    float dn = 0;
    float w = 0;

    for (int i = -pad; i <= pad; ++i)
    {
      for (int j = -pad; j <= pad; ++j)
      {
        const float dk = shared[(ty + i) * resolution + (tx + j)];
        const float delta = d0 - dk;

        const float wr = expf(-Vector2f(i, j).SquaredNorm() / (pad * pad));
        const float ws = expf(-(delta * delta) / 0.0004f);
        const float ww = wr * ws;
        dn += ww * dk;
        w += ww;
      }
    }

    const int output = y * image_width + x;
    dst[output] = dn / w;
  }
}

void ComputeNormals(const float* depths, const Projection& projection,
    Vector3f* normals, int image_width, int image_height)
{
  const dim3 threads(16, 16);
  const dim3 total(image_width, image_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(ComputeNormalsKernel<16>, blocks, threads, 0, 0, depths,
      projection, normals, image_width, image_height);
}

void FilterDepths(int image_width, int image_height, const float* src,
    float* dst)
{
  const dim3 threads(16, 16);
  const dim3 total(image_width, image_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(FilterDepthsKernel<16>, blocks, threads, 0, 0, image_width,
      image_height, src, dst);
}

} // namespace vulcan