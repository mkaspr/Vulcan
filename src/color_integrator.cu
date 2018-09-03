#include <vulcan/color_integrator.h>
#include <vulcan/block.h>
#include <vulcan/buffer.h>
#include <vulcan/device.h>
#include <vulcan/exception.h>
#include <vulcan/frame.h>
#include <vulcan/hash.h>
#include <vulcan/math.h>
#include <vulcan/volume.h>
#include <vulcan/voxel.h>

namespace vulcan
{
namespace
{

VULCAN_GLOBAL
void IntegrateDepthKernel(const int* indices, const HashEntry* hash_entries,
    float voxel_length, float block_length, float truncation_length,
    float min_depth, float max_depth, const float* depths,
    int image_width, int image_height, float max_dist_weight,
    const Projection projection, const Transform Tdw, Voxel* voxels)
{
  // get voxel indices
  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const int z = threadIdx.z;

  // get block hash table entry
  const int entry_index = indices[blockIdx.x];
  const HashEntry entry = hash_entries[entry_index];
  const Block& block = entry.block;

  // compute voxel point in world frame
  const Vector3s& block_index = block.GetOrigin();
  const Vector3f block_offset = block_length * Vector3f(block_index);
  const Vector3f voxel_offset = voxel_length * (Vector3f(x, y, z) + 0.5f);
  const Vector3f Xwp = block_offset + voxel_offset;

  // convert point to camera frame
  const Vector3f Xdp = Vector3f(Tdw * Vector4f(Xwp, 1));

  // project point to image plane
  const Vector2f uv = projection.Project(Xdp);

  // check if point inside image
  if (uv[0] >= 0 && uv[0] < image_width && uv[1] >= 0 && uv[1] < image_height)
  {
    // get measurement from depth image
    const int image_index = int(uv[1]) * image_width + int(uv[0]);
    const float depth = depths[image_index];

    // ignore invalid depth values
    if (depth < min_depth || depth > max_depth) return;

    // compute signed distance
    const float distance = depth - Xdp[2];

    // check if within truncated segment
    if (distance > -truncation_length)
    {
      // compute voxel index
      const int r = Block::resolution;
      const int r2 = Block::resolution * Block::resolution;
      const int block_index = entry.data * Block::voxel_count;
      const int voxel_index = block_index + z * r2 + y * r + x;
      Voxel voxel = voxels[voxel_index];

      // update voxel distance
      const float prev_dist = voxel.distance_weight * voxel.distance;
      const float curr_dist = min(1.0f, distance / truncation_length);
      const float dist_weight = voxel.distance_weight + 1;
      voxel.distance_weight = min(max_dist_weight, dist_weight);
      voxel.distance = (prev_dist + curr_dist) / dist_weight;

      // store updates in global memory
      voxels[voxel_index] = voxel;
    }
  }
}

VULCAN_GLOBAL
void IntegrateColorKernel(const int* indices, const HashEntry* hash_entries,
    float voxel_length, float block_length, const Vector3f* colors,
    int image_width, int image_height, float max_color_weight,
    const Projection projection, const Transform Tcw, Voxel* voxels)
{
  // get voxel indices
  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const int z = threadIdx.z;

  // get block hash table entry
  const int entry_index = indices[blockIdx.x];
  const HashEntry entry = hash_entries[entry_index];
  const Block& block = entry.block;

  // compute voxel point in world frame
  const Vector3s& block_index = block.GetOrigin();
  const Vector3f block_offset = block_length * Vector3f(block_index);
  const Vector3f voxel_offset = voxel_length * (Vector3f(x, y, z) + 0.5f);
  const Vector3f Xwp = block_offset + voxel_offset;

  // convert point to camera frame
  const Vector3f Xcp = Vector3f(Tcw * Vector4f(Xwp, 1));

  // project point to image plane
  const Vector2f uv = projection.Project(Xcp);

  // check if point inside image
  if (uv[0] >= 0 && uv[0] < image_width && uv[1] >= 0 && uv[1] < image_height)
  {
    // compute voxel index
    const int r = Block::resolution;
    const int r2 = Block::resolution * Block::resolution;
    const int block_index = entry.data * Block::voxel_count;
    const int voxel_index = block_index + z * r2 + y * r + x;
    Voxel voxel = voxels[voxel_index];

    // check if voxel near surface
    if (fabsf(voxel.distance) < 1.0) // TODO: expose parameter
    {
      // update voxel color
      const int image_index = int(uv[1]) * image_width + int(uv[0]);
      const Vector3f prev_color = voxel.color_weight * voxel.GetColor();
      const Vector3f curr_color = colors[image_index];
      const float color_weight = voxel.color_weight + 1;
      voxel.color_weight = min(max_color_weight, color_weight);
      voxel.SetColor((prev_color + curr_color) / color_weight);

      // store updates in global memory
      voxels[voxel_index] = voxel;
    }
  }
}

} // namespace

ColorIntegrator::ColorIntegrator(std::shared_ptr<Volume> volume) :
  Integrator(volume)
{
}

void ColorIntegrator::Integrate(const Frame& frame)
{
  IntegrateDepth(frame);
  IntegrateColor(frame);
}

void ColorIntegrator::IntegrateDepth(const Frame& frame)
{
  const Buffer<int>& index_buffer = volume_->GetVisibleBlocks();
  const Buffer<HashEntry>& entry_buffer = volume_->GetHashEntries();
  Buffer<Voxel>& voxel_buffer = volume_->GetVoxels();

  const int* indices = index_buffer.GetData();
  const HashEntry* entries = entry_buffer.GetData();
  const float voxel_length = volume_->GetVoxelLength();
  const float block_length = Block::resolution * voxel_length;
  const float truncation_length = volume_->GetTruncationLength();
  const float* depths = frame.depth_image->GetData();
  const int image_width = frame.depth_image->GetWidth();
  const int image_height = frame.depth_image->GetHeight();
  const Transform Tdw = frame.depth_to_world_transform.Inverse();
  const Projection& projection = frame.depth_projection;
  Voxel* voxels = voxel_buffer.GetData();

  const int resolution = Block::resolution;
  const size_t blocks = index_buffer.GetSize();
  const dim3 threads(resolution, resolution, resolution);

  CUDA_LAUNCH(IntegrateDepthKernel, blocks, threads, 0, 0, indices, entries,
      voxel_length, block_length, truncation_length, depth_range_[0],
      depth_range_[1], depths, image_width, image_height, max_distance_weight_,
      projection, Tdw, voxels);
}

void ColorIntegrator::IntegrateColor(const Frame& frame)
{
  const Buffer<int>& index_buffer = volume_->GetVisibleBlocks();
  const Buffer<HashEntry>& entry_buffer = volume_->GetHashEntries();
  Buffer<Voxel>& voxel_buffer = volume_->GetVoxels();

  const int* indices = index_buffer.GetData();
  const HashEntry* entries = entry_buffer.GetData();
  const float voxel_length = volume_->GetVoxelLength();
  const float block_length = Block::resolution * voxel_length;
  const Vector3f* colors = frame.color_image->GetData();
  const int image_width = frame.color_image->GetWidth();
  const int image_height = frame.color_image->GetHeight();
  const Transform& Twd = frame.depth_to_world_transform;
  const Transform& Tcd = frame.depth_to_color_transform;
  const Transform Tcw = Tcd * Twd.Inverse();
  const Projection& projection = frame.color_projection;
  Voxel* voxels = voxel_buffer.GetData();

  const int resolution = Block::resolution;
  const size_t blocks = index_buffer.GetSize();
  const dim3 threads(resolution, resolution, resolution);

  CUDA_LAUNCH(IntegrateColorKernel, blocks, threads, 0, 0, indices, entries,
      voxel_length, block_length, colors, image_width, image_height,
      max_color_weight_, projection, Tcw, voxels);
}

} // namespace vulcan