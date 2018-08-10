#include <vulcan/sampler.h>
#include <vulcan/device.h>
#include <vulcan/image.h>

namespace vulcan
{

template <Sampler::Filter filter, typename T>
VULCAN_GLOBAL
void GetSubimageKernel(const T* image, T* subimage, int width, int height)
{
  const int sub_width = width / 2;
  const int sub_height = height / 2;
  const int sub_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int sub_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (sub_x < sub_width && sub_y < sub_height)
  {
    T value;
    const int x = 2 * sub_x;
    const int y = 2 * sub_y;

    if (filter == Sampler::FILTER_LINEAR)
    {
      value  = image[width * (y + 0) + (x + 0)];
      value += image[width * (y + 0) + (x + 1)];
      value += image[width * (y + 1) + (x + 0)];
      value += image[width * (y + 1) + (x + 1)];
      value *= 0.25f;
    }
    else
    {
      value = image[y * width + x];
    }

    subimage[sub_y * sub_width + sub_x] = value;
  }
}

template <typename T>
VULCAN_GLOBAL
void GetGradientKernel(const T* image, T* gx, T* gy, int width, int height)
{
  const int shared_resolution = 18;
  const int shared_size = shared_resolution * shared_resolution;
  VULCAN_SHARED T shared[shared_size];

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int block_size = blockDim.x * blockDim.y;
  const int index = y * width + x;

  int shared_index = threadIdx.y * blockDim.x + threadIdx.x;
  const int global_offset_x = blockIdx.x * blockDim.x - 1;
  const int global_offset_y = blockIdx.y * blockDim.y - 1;

  do
  {
    int xx = global_offset_x + (shared_index % shared_resolution);
    int yy = global_offset_y + (shared_index / shared_resolution);

    xx = clamp(xx, 0, width - 1);
    yy = clamp(yy, 0, height - 1);

    const int global_index = yy * width + xx;
    shared[shared_index] = image[global_index];
    shared_index += block_size;
  }
  while (shared_index < shared_size);

  __syncthreads();

  if (x < width && y < height)
  {
    T xx;
    T yy;

    const int sx = threadIdx.x + 1;
    const int sy = threadIdx.y + 1;

    xx  = 1 * shared[shared_resolution * (sy - 1) + (sx + 1)];
    xx += 2 * shared[shared_resolution * (sy + 0) + (sx + 1)];
    xx += 1 * shared[shared_resolution * (sy + 1) + (sx + 1)];

    xx -= 1 * shared[shared_resolution * (sy - 1) + (sx - 1)];
    xx -= 2 * shared[shared_resolution * (sy + 0) + (sx - 1)];
    xx -= 1 * shared[shared_resolution * (sy + 1) + (sx - 1)];

    yy  = 1 * shared[shared_resolution * (sy + 1) + (sx - 1)];
    yy += 2 * shared[shared_resolution * (sy + 1) + (sx + 0)];
    yy += 1 * shared[shared_resolution * (sy + 1) + (sx + 1)];

    yy -= 1 * shared[shared_resolution * (sy - 1) + (sx - 1)];
    yy -= 2 * shared[shared_resolution * (sy - 1) + (sx + 0)];
    yy -= 1 * shared[shared_resolution * (sy - 1) + (sx + 1)];

    gx[index] = 0.125f * xx;
    gy[index] = 0.125f * yy;
  }
}

template <typename T>
inline void GetSubimage(const T& image, T& subimage, Sampler::Filter filter)
{
  const int w = image.GetWidth();
  const int h = image.GetHeight();
  subimage.Resize(w / 2, h / 2);

  const dim3 total(w, h);
  const dim3 threads(16, 16);
  const dim3 blocks = GetKernelBlocks(total, threads);

  switch (filter)
  {
    case Sampler::FILTER_NEAREST:
    {
      CUDA_LAUNCH(GetSubimageKernel<Sampler::FILTER_NEAREST>,
          blocks, threads, 0, 0, image.GetData(), subimage.GetData(), w, h);

      break;
    }

    case Sampler::FILTER_LINEAR:
    {
      CUDA_LAUNCH(GetSubimageKernel<Sampler::FILTER_LINEAR>,
          blocks, threads, 0, 0, image.GetData(), subimage.GetData(), w, h);

      break;
    }
  }
}

template <typename T>
void GetGradient(const T& image, T& gradient_x, T& gradient_y)
{
  const int w = image.GetWidth();
  const int h = image.GetHeight();
  gradient_x.Resize(w, h);
  gradient_y.Resize(w, h);

  const dim3 total(w, h);
  const dim3 threads(16, 16);
  const dim3 blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(GetGradientKernel, blocks, threads, 0, 0, image.GetData(),
      gradient_x.GetData(), gradient_y.GetData(), w, h);
}

Sampler::Sampler()
{
}

void Sampler::GetSubimage(const Image& image, Image& subimage,
    Filter filter) const
{
  vulcan::GetSubimage(image, subimage, filter);
}

void Sampler::GetSubimage(const ColorImage& image, ColorImage& subimage,
    Filter filter) const
{
  vulcan::GetSubimage(image, subimage, filter);
}

void Sampler::GetGradient(const Image& image, Image& gradient_x,
    Image& gradient_y) const
{
  vulcan::GetGradient(image, gradient_x, gradient_y);
}

void Sampler::GetGradient(const ColorImage& image, ColorImage& gradient_x,
    ColorImage& gradient_y) const
{
  vulcan::GetGradient(image, gradient_x, gradient_y);
}

} // namespace vulcan