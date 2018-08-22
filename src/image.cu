#include <vulcan/image.h>
#include <vulcan/device.h>

namespace vulcan
{

template <bool nearest>
VULCAN_GLOBAL
void DownsampleKernel(int src_w, int src_h, const float* src,
    int dst_w, int dst_h, float* dst)
{
  const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (dst_x < dst_w && dst_y < dst_h)
  {
    float sample = 0;
    const int src_x = 2 * dst_x;
    const int src_y = 2 * dst_y;

    if (nearest)
    {
      sample = src[src_y * src_w + src_x];
    }
    else
    {
      const int src_x = 2 * dst_x;
      const int src_y = 2 * dst_y;
      sample += src[(src_y + 0) * src_w + (src_x + 1)];
      sample += src[(src_y + 0) * src_w + (src_x + 0)];
      sample += src[(src_y + 1) * src_w + (src_x + 1)];
      sample += src[(src_y + 1) * src_w + (src_x + 0)];
      sample *= 0.25f;
    }

    dst[dst_y * dst_w + dst_x] = sample;
  }
}

template <bool nearest>
VULCAN_GLOBAL
void DownsampleKernel(int src_w, int src_h, const Vector3f* src,
    int dst_w, int dst_h, Vector3f* dst)
{
  const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (dst_x < dst_w && dst_y < dst_h)
  {
    Vector3f sample(0, 0, 0);
    const int src_x = 2 * dst_x;
    const int src_y = 2 * dst_y;

    if (nearest)
    {
      sample = src[src_y * src_w + src_x];
    }
    else
    {
      const int src_x = 2 * dst_x;
      const int src_y = 2 * dst_y;
      sample += src[(src_y + 0) * src_w + (src_x + 1)];
      sample += src[(src_y + 0) * src_w + (src_x + 0)];
      sample += src[(src_y + 1) * src_w + (src_x + 1)];
      sample += src[(src_y + 1) * src_w + (src_x + 0)];
      sample *= 0.25f;
    }

    dst[dst_y * dst_w + dst_x] = sample;
  }
}

void Image::Downsample(Image& image, bool nearest) const
{
  VULCAN_DEBUG_MSG(size_[0] % 2 == 0 && size_[1] % 2 == 0,
      "even image dimensions required");

  image.Resize(size_ / 2);
  const float* src = data_;
  float* dst = image.GetData();

  const int src_w = size_[0];
  const int src_h = size_[1];
  const int dst_w = image.GetWidth();
  const int dst_h = image.GetHeight();

  const dim3 threads(16, 16);
  const dim3 total(dst_w, dst_h);
  const dim3 blocks = GetKernelBlocks(total, threads);

  if (nearest)
  {
    CUDA_LAUNCH(DownsampleKernel<true>, blocks, threads, 0, 0, src_w, src_h,
        src, dst_w, dst_h, dst);
  }
  else
  {
    CUDA_LAUNCH(DownsampleKernel<false>, blocks, threads, 0, 0, src_w, src_h,
        src, dst_w, dst_h, dst);
  }
}

void ColorImage::Downsample(ColorImage& image, bool nearest) const
{
  VULCAN_DEBUG_MSG(size_[0] % 2 == 0 && size_[1] % 2 == 0,
      "even image dimensions required");

  image.Resize(size_ / 2);
  const Vector3f* src = data_;
  Vector3f* dst = image.GetData();

  const int src_w = size_[0];
  const int src_h = size_[1];
  const int dst_w = image.GetWidth();
  const int dst_h = image.GetHeight();

  const dim3 threads(16, 16);
  const dim3 total(dst_w, dst_h);
  const dim3 blocks = GetKernelBlocks(total, threads);

  if (nearest)
  {
    CUDA_LAUNCH(DownsampleKernel<true>, blocks, threads, 0, 0, src_w, src_h,
        src, dst_w, dst_h, dst);
  }
  else
  {
    CUDA_LAUNCH(DownsampleKernel<false>, blocks, threads, 0, 0, src_w, src_h,
        src, dst_w, dst_h, dst);
  }
}

} // namespace vulcan