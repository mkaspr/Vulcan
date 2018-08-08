#include <vulcan/tracer.h>
#include <vulcan/tracer.cuh>
#include <vulcan/block.h>
#include <vulcan/device.h>
#include <vulcan/exception.h>
#include <vulcan/frame.h>
#include <vulcan/math.h>
#include <vulcan/volume.h>

namespace vulcan
{

Tracer::Tracer(std::shared_ptr<const Volume> volume) :
  volume_(volume)
{
  Initialize();
}

std::shared_ptr<const Volume> Tracer::GetVolume() const
{
  return volume_;
}

void Tracer::Trace(Frame& frame)
{
  ComputePatches(frame);
  ComputeBounds(frame);
  ComputePoints(frame);
  ComputeNormals(frame);
}

void Tracer::ComputePatches(const Frame& frame)
{
  const Buffer<int>& visible = volume_->GetVisibleBlocks();
  const Buffer<HashEntry>& entries = volume_->GetHashEntries();
  const float block_length = Block::resolution * volume_->GetVoxelLength();
  const int image_height = frame.depth_image->GetHeight();
  const int image_width = frame.depth_image->GetWidth();
  const int bounds_width = 80; // TODO: expose variable
  const int bounds_height = 60; // TODO: expose variable

  ResetBufferSize();

  vulcan::ComputePatches(visible.GetData(), entries.GetData(), frame.Tcw,
      frame.projection, block_length, visible.GetSize(), image_width,
      image_height, bounds_width, bounds_height, patches_.GetData(),
      buffer_size_.GetData());

  patches_.Resize(GetBufferSize());
}

void Tracer::ComputeBounds(const Frame& frame)
{
  ResetBoundsBuffer();
  const int total = patches_.GetSize();
  vulcan::ComputeBounds(patches_.GetData(), bounds_.GetData(), 80, total);
}

void Tracer::ComputePoints(Frame& frame)
{
  const Buffer<HashEntry>& entries = volume_->GetHashEntries();
  const Buffer<Voxel>& voxels = volume_->GetVoxels();
  const float trunc_length = volume_->GetTruncationLength();
  const int w = frame.depth_image->GetWidth();
  const int h = frame.depth_image->GetHeight();
  const int block_count = volume_->GetMainBlockCount();
  const float voxel_length = volume_->GetVoxelLength();
  const float block_length = Block::resolution * voxel_length;
  frame.color_image->Resize(w, h);

  vulcan::ComputePoints(entries.GetData(), voxels.GetData(), bounds_.GetData(),
      block_count, block_length, voxel_length, trunc_length, frame.Tcw,
      frame.projection, frame.depth_image->GetData(),
      frame.color_image->GetData(), w, h, 80, 60);
}

void Tracer::ComputeNormals(Frame& frame)
{
  const float* depths = frame.depth_image->GetData();
  Vector3f* normals = frame.normal_image->GetData();
  const int image_width = frame.depth_image->GetWidth();
  const int image_height = frame.depth_image->GetHeight();
  const Projection& proj = frame.projection;
  vulcan::ComputeNormals(depths, proj, normals, image_width, image_height);
}

void Tracer::ResetBoundsBuffer()
{
  vulcan::ResetBoundsBuffer(bounds_.GetData(), bounds_.GetSize());
}

void Tracer::ResetBufferSize()
{
  const int value = 0;
  const size_t bytes = sizeof(value);
  const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  CUDA_DEBUG(cudaMemcpy(buffer_size_.GetData(), &value, bytes, kind));
}

int Tracer::GetBufferSize()
{
  int value = 0;
  const size_t bytes = sizeof(value);
  const cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
  CUDA_DEBUG(cudaMemcpy(&value, buffer_size_.GetData(), bytes, kind));
  return value;
}

void Tracer::Initialize()
{
  CreatePatchBuffer();
  CreateBoundsBuffer();
  CreateSizeBuffer();
}

void Tracer::CreatePatchBuffer()
{
  // TODO: size according to system
  patches_.Reserve(262144);
}

void Tracer::CreateBoundsBuffer()
{
  // TODO: size according to system
  bounds_.Resize(80 * 60);
}

void Tracer::CreateSizeBuffer()
{
  buffer_size_.Resize(1);
}

} // namespace vulcan