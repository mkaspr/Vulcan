#pragma once

#include <vulcan/device.h>

namespace vulcan
{

template <typename T>
class Buffer
{
  public:

    Buffer() :
      data_(nullptr),
      capacity_(0),
      size_(0)
    {
    }

    Buffer(size_t size) :
      data_(nullptr),
      capacity_(0),
      size_(0)
    {
      Resize(size);
    }

    ~Buffer()
    {
      cudaFree(data_);
    }

    inline const T* GetData() const
    {
      return data_;
    }

    inline T* GetData()
    {
      return data_;
    }

    inline size_t GetSize() const
    {
      return size_;
    }

    inline size_t GetCapacity() const
    {
      return capacity_;
    }

    inline size_t IsEmpty() const
    {
      return size_ == 0;
    }

    inline void Resize(size_t size)
    {
      if (size > capacity_) Reserve(size);
      size_ = size;
    }

    void Reserve(size_t capacity)
    {
      if (capacity > capacity_)
      {
        CUDA_DEBUG(cudaFree(data_));
        const size_t bytes = sizeof(T) * capacity;
        CUDA_DEBUG(cudaMalloc(&data_, bytes));
        capacity_ = capacity;
      }
    }

    void CopyFromDevice(T* buffer)
    {
      const size_t bytes = sizeof(T) * size_;
      const cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
      CUDA_DEBUG(cudaMemcpy(data_, buffer, bytes, kind));
    }

    void CopyToDevice(T* buffer) const
    {
      const size_t bytes = sizeof(T) * size_;
      const cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
      CUDA_DEBUG(cudaMemcpy(buffer, data_, bytes, kind));
    }

    void CopyFromHost(const T* buffer)
    {
      const size_t bytes = sizeof(T) * size_;
      const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
      CUDA_DEBUG(cudaMemcpy(data_, buffer, bytes, kind));
    }

    void CopyToHost(T* buffer) const
    {
      const size_t bytes = sizeof(T) * size_;
      const cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
      CUDA_DEBUG(cudaMemcpy(buffer, data_, bytes, kind));
    }

  private:

    Buffer(const Buffer& buffer);

    Buffer& operator=(const Buffer& buffer);

  protected:

    T* data_;

    size_t capacity_;

    size_t size_;
};

} // namespace vulcan