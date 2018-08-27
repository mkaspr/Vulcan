#pragma once

#include <opencv2/opencv.hpp>
#include <vulcan/device.h>
#include <vulcan/exception.h>
#include <vulcan/matrix.h>

namespace vulcan
{

class Image
{
  public:

    VULCAN_HOST_DEVICE
    Image() :
      size_(0, 0),
      data_(nullptr)
    {
    }

    VULCAN_HOST
    Image(int w, int h) :
      size_(0, 0),
      data_(nullptr)
    {
      Resize(w, h);
    }

    VULCAN_HOST
    ~Image()
    {
      cudaFree(data_);
    }

    VULCAN_HOST_DEVICE
    inline int GetWidth() const
    {
      return size_[0];
    }

    VULCAN_HOST_DEVICE
    inline int GetHeight() const
    {
      return size_[1];
    }

    VULCAN_HOST_DEVICE
    inline const Vector2i& GetSize() const
    {
      return size_;
    }

    VULCAN_HOST_DEVICE
    inline int GetTotal() const
    {
      return size_[0] * size_[1];
    }

    VULCAN_HOST_DEVICE
    inline int GetBytes() const
    {
      return sizeof(float) * GetTotal();
    }

    VULCAN_HOST_DEVICE
    inline const float* GetData() const
    {
      return data_;
    }

    VULCAN_HOST_DEVICE
    inline float* GetData()
    {
      return data_;
    }

    VULCAN_HOST
    void Resize(int w, int h)
    {
      Resize(Vector2i(w, h));
    }

    VULCAN_HOST
    void Resize(const Vector2i& size)
    {
      VULCAN_DEBUG(size[0] >= 0 && size[1] >= 0);
      const int old_total = GetTotal();
      size_ = size;
      const int new_total = GetTotal();

      if (new_total != old_total)
      {
        CUDA_DEBUG(cudaFree(data_));
        CUDA_DEBUG(cudaMalloc(&data_, GetBytes()));
      }
    }

    VULCAN_HOST
    void Load(const std::string& file, float scale = 1)
    {
      cv::Mat image = cv::imread(file, CV_LOAD_IMAGE_ANYDEPTH);
      VULCAN_ASSERT_MSG(image.data != nullptr, "unable to load file");
      if (image.channels() == 3) cv::cvtColor(image, image, CV_RGB2GRAY);
      image.convertTo(image, CV_32FC1, scale);
      Resize(image.cols, image.rows);
      const size_t bytes = sizeof(float) * image.total();
      const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
      CUDA_DEBUG(cudaMemcpy(data_, image.data, bytes, kind));
    }

    VULCAN_HOST
    void GetGradients(Image& gx, Image& gy) const;

    VULCAN_HOST
    void Downsample(Image& image, bool nearest) const;

  protected:

    Vector2i size_;

    float* data_;
};

class ColorImage
{
  public:

    VULCAN_HOST_DEVICE
    ColorImage() :
      size_(0, 0),
      data_(nullptr)
    {
    }

    VULCAN_HOST
    ColorImage(int w, int h) :
      size_(0, 0),
      data_(nullptr)
    {
      Resize(w, h);
    }

    VULCAN_HOST
    ~ColorImage()
    {
      cudaFree(data_);
    }

    VULCAN_HOST_DEVICE
    inline int GetWidth() const
    {
      return size_[0];
    }

    VULCAN_HOST_DEVICE
    inline int GetHeight() const
    {
      return size_[1];
    }

    VULCAN_HOST_DEVICE
    inline const Vector2i& GetSize() const
    {
      return size_;
    }

    VULCAN_HOST_DEVICE
    inline int GetTotal() const
    {
      return size_[0] * size_[1];
    }

    VULCAN_HOST_DEVICE
    inline int GetBytes() const
    {
      return sizeof(Vector3f) * GetTotal();
    }

    VULCAN_HOST_DEVICE
    inline const Vector3f* GetData() const
    {
      return data_;
    }

    VULCAN_HOST_DEVICE
    inline Vector3f* GetData()
    {
      return data_;
    }

    VULCAN_HOST
    void Resize(int w, int h)
    {
      Resize(Vector2i(w, h));
    }

    VULCAN_HOST
    void Resize(const Vector2i& size)
    {
      VULCAN_DEBUG(size[0] >= 0 && size[1] >= 0);
      const int old_total = GetTotal();
      size_ = size;
      const int new_total = GetTotal();

      if (new_total != old_total)
      {
        CUDA_DEBUG(cudaFree(data_));
        CUDA_DEBUG(cudaMalloc(&data_, GetBytes()));
      }
    }

    VULCAN_HOST
    void Load(const std::string& file, float scale = 1)
    {
      cv::Mat image = cv::imread(file, CV_LOAD_IMAGE_ANYCOLOR);
      VULCAN_ASSERT_MSG(image.data != nullptr, "unable to load file");
      if (image.channels() == 1) cv::cvtColor(image, image, CV_GRAY2RGB);
      image.convertTo(image, CV_32FC3, scale);
      cv::cvtColor(image, image, CV_BGR2RGB);
      Resize(image.cols, image.rows);
      const size_t bytes = sizeof(Vector3f) * image.total();
      const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
      CUDA_DEBUG(cudaMemcpy(data_, image.data, bytes, kind));
    }

    VULCAN_HOST
    void ConvertTo(Image& image) const;

    VULCAN_HOST
    void Downsample(ColorImage& image, bool nearest) const;

  protected:

    Vector2i size_;

    Vector3f* data_;
};

} // namespace vulcan