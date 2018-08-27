#include <vulcan/image.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <vulcan/device.h>

namespace vulcan
{

VULCAN_GLOBAL
void ConvertKernel(int total, const Vector3f* src, float* dst)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < total)
  {
    const Vector3f value = src[index];
    dst[index] = (value[0] + value[1] + value[2]) / 3.0f;
  }
}

template <int BLOCK_DIM>
VULCAN_GLOBAL
void GetGradientsKernel(int width, int height, const float* values,
    float* x_gradients, float* y_gradients)
{
  // allocate shared memory
  const int buffer_dim = BLOCK_DIM + 2;
  const int buffer_size = buffer_dim * buffer_dim;
  VULCAN_SHARED float buffer[buffer_size];

  // get launch indices
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  int sindex = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_size = blockDim.x * blockDim.y;

  // copy image patch to shared memory
  do
  {
    // initialize default value
    float value = 0;

    // get source image indices
    const int vx = (blockIdx.x * blockDim.x - 1) + (sindex % buffer_dim);
    const int vy = (blockIdx.y * blockDim.y - 1) + (sindex / buffer_dim);

    // check if within image bounds
    if (vx >= 0 && vx < width && vy >= 0 && vy < height)
    {
      // read value from global memory
      value = values[vy * width + vx];
    }

    // store value in shared memory
    buffer[sindex] = value;

    // advance to next shared index
    sindex += block_size;
  }
  while (sindex < buffer_size);

  // wait for all threads to finish
  __syncthreads();

  // check if current thread within image bounds
  if (x < width && y < height)
  {
    // initialize default gradients
    float gx = 0;
    float gy = 0;

    // get kernel top-left indices in shared memory
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // read top row values from shared memory
    const float i00 = 0.125f * buffer[(ty + 0) * buffer_dim + (tx + 0)];
    const float i01 = 0.250f * buffer[(ty + 0) * buffer_dim + (tx + 1)];
    const float i02 = 0.125f * buffer[(ty + 0) * buffer_dim + (tx + 2)];

    // read center row values from shared memory
    const float i10 = 0.250f * buffer[(ty + 1) * buffer_dim + (tx + 0)];
    const float i12 = 0.250f * buffer[(ty + 1) * buffer_dim + (tx + 2)];

    // read bottom row values from shared memory
    const float i20 = 0.125f * buffer[(ty + 2) * buffer_dim + (tx + 0)];
    const float i21 = 0.250f * buffer[(ty + 2) * buffer_dim + (tx + 1)];
    const float i22 = 0.125f * buffer[(ty + 2) * buffer_dim + (tx + 2)];

    // compute gradient values
    gx = (i02 + i12 + i22) - (i00 + i10 + i20);
    gy = (i20 + i21 + i22) - (i00 + i01 + i02);

    // store result in global memory
    const int index = y * width + x;
    x_gradients[index] = gx;
    y_gradients[index] = gy;
  }
}

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

void Image::GetGradients(Image& gx, Image& gy) const
{
  gx.Resize(size_);
  gy.Resize(size_);

  float* gx_data = gx.GetData();
  float* gy_data = gy.GetData();

  const dim3 threads(16, 16);
  const dim3 total(size_[0], size_[1]);
  const dim3 blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(GetGradientsKernel<16>, blocks, threads, 0, 0,
      size_[0], size_[1], data_, gx_data, gy_data);
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

void Image::Save(const std::string& file, int type, float alpha,
    float beta) const
{
  thrust::device_ptr<const float> d_data(data_);
  thrust::host_vector<float> data(d_data, d_data + GetTotal());
  cv::Mat image(size_[1], size_[0], CV_32FC1, data.data());
  image.convertTo(image, type, alpha, beta);
  cv::imwrite(file, image);
}

void ColorImage::ConvertTo(Image& image) const
{
  image.Resize(size_);
  const int threads = 512;
  const int total = GetTotal();
  const int blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(ConvertKernel, blocks, threads, 0, 0, total, data_,
      image.GetData());
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

void ColorImage::Save(const std::string& file, int type, float alpha,
    float beta) const
{
  thrust::device_ptr<const Vector3f> d_data(data_);
  thrust::host_vector<Vector3f> data(d_data, d_data + GetTotal());
  cv::Mat image(size_[1], size_[0], CV_32FC3, data.data());
  image.convertTo(image, type, alpha, beta);
  cv::cvtColor(image, image, CV_RGB2BGR);
  cv::imwrite(file, image);
}

} // namespace vulcan