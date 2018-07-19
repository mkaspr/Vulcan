#include <vulcan/detector.h>
#include <limits>
#include <vulcan/device.h>
#include <vulcan/exception.h>
#include <vulcan/util.cuh>

namespace vulcan
{

VULCAN_DEVICE int buffer_size;

template <int BLOCK_SIZE>
VULCAN_GLOBAL
void FilterKernel(const Vector3f* points, float radius, Vector3f origin,
    Vector2f bx, Vector2f by, Vector2f bz, Vector3f* output, int count)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  Vector3f point = Vector3f::Zeros();
  bool valid = false;

  if (index < count)
  {
    point = points[index];

    valid = ((radius <= 0 || (point - origin).Norm() < radius) &&
        (bx[0] > bx[1] || (point[0] >= bx[0] && point[0] <= bx[1])) &&
        (by[0] > by[1] || (point[0] >= by[0] && point[0] <= by[1])) &&
        (bz[0] > bz[1] || (point[0] >= bz[0] && point[0] <= bz[1])));
  }

  const int offset = PrefixSum<BLOCK_SIZE>(valid, threadIdx.x, buffer_size);
  if (offset >= 0) output[offset] = point;
}

template <int BLOCK_SIZE>
VULCAN_GLOBAL
void RemovalKernel(Vector3f* points, Vector3f center, float max_dist, int count)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  Vector3f point = Vector3f::Zeros();
  bool valid = false;

  if (index < count)
  {
    point = points[index];
    valid = (point - center).Norm() <= max_dist;
  }

  const int offset = PrefixSum<BLOCK_SIZE>(valid, threadIdx.x, buffer_size);
  if (offset >= 0) points[offset] = point;
}

VULCAN_GLOBAL
void DistanceKernel(const Vector3f* a, const Vector3f b, float* c, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    c[i] = (a[i] - b).Norm();
  }
}

Detector::Detector() :
  radius_(2.0),
  origin_(0, 0, 0),
  min_inlier_count_(100)
{
  Initialize();
}

Detector::~Detector()
{
  cublasDestroy(handle_);
}

float Detector::GetRadius() const
{
  return radius_;
}

void Detector::SetRadius(float radius)
{
  radius_ = radius;
}

const Vector3f& Detector::GetOrigin() const
{
  return origin_;
}

void Detector::SetOrigin(const Vector3f& origin)
{
  origin_ = origin;
}

const Vector2f& Detector::GetBounds(int axis) const
{
  VULCAN_DEBUG(axis >= 0 && axis < 3);
  return bounds_[axis];
}

void Detector::SetBounds(int axis, const Vector2f& bounds)
{
  VULCAN_DEBUG(axis >= 0 && axis < 3);
  bounds_[axis] = bounds;
}

int Detector::GetMinInlierCount() const
{
  return min_inlier_count_;
}

void Detector::SetMinInlierCount(int count)
{
  VULCAN_DEBUG(count > 0);
  min_inlier_count_ = count;
}

Vector3f Detector::Detect(const Buffer<Vector3f>& points)
{
  Filter(points);
  return (BoxDetected()) ? GetValidPosition() : GetInvalidPosition();
}

bool Detector::BoxDetected() const
{
  return int(points_.GetSize()) >= min_inlier_count_;
}

Vector3f Detector::GetValidPosition() const
{
  Vector3f result;
  const int count = points_.GetSize();
  const Vector3f* points = points_.GetData();
  const float* data = reinterpret_cast<const float*>(points);
  cublasSasum(handle_, count, data + 0, 3, &result[0]);
  cublasSasum(handle_, count, data + 1, 3, &result[1]);
  cublasSasum(handle_, count, data + 2, 3, &result[2]);
  return result / count;
}

Vector3f Detector::GetInvalidPosition() const
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  return Vector3f(nan, nan, nan);
}

void Detector::Filter(const Buffer<Vector3f>& points)
{
  const int threads = 512;
  const int total = points.GetSize();
  const int blocks = GetKernelBlocks(total, threads);
  const Vector3f* data = points.GetData();
  points_.Resize(points.GetSize());
  Vector3f* output = points_.GetData();

  ResetBufferSize();

  CUDA_LAUNCH(FilterKernel<512>, blocks, threads, 0, 0, data, radius_, origin_,
      bounds_[0], bounds_[1], bounds_[2], output, total);

  points_.Resize(GetBufferSize());

  if (!points_.IsEmpty())
  {
    const int threads = 512;
    const int total = points_.GetSize();
    const int blocks = GetKernelBlocks(total, threads);
    const Vector3f center = GetValidPosition();
    const Vector3f* a = points_.GetData();
    distances_.Resize(total);
    float* c = distances_.GetData();
    CUDA_LAUNCH(DistanceKernel, blocks, threads, 0, 0, a, center, c, total);

    float squared_error;
    cublasSdot(handle_, total, c, 1, c, 1, &squared_error);
    const float stdev = std::sqrt(squared_error / total);
    const float limit = 1.5f * stdev;

    ResetBufferSize();

    CUDA_LAUNCH(RemovalKernel<512>, blocks, threads, 0, 0, output, center,
        limit, total);

    points_.Resize(GetBufferSize());
  }
}

int Detector::GetBufferSize() const
{
  int result;
  const size_t bytes = sizeof(int);
  CUDA_ASSERT(cudaMemcpyFromSymbol(&result, buffer_size, bytes));
  return result;
}

void Detector::ResetBufferSize() const
{
  const int result = 0;
  const size_t bytes = sizeof(int);
  CUDA_ASSERT(cudaMemcpyToSymbol(buffer_size, &result, bytes));
}

void Detector::Initialize()
{
  CreateHandle();
  CreateBounds();
}

void Detector::CreateHandle()
{
  cublasCreate(&handle_);
}

void Detector::CreateBounds()
{
  bounds_[0] = Vector2f(1, -1);
  bounds_[1] = Vector2f(1, -1);
  bounds_[2] = Vector2f(1, -1);
}

} // namespace vulcan