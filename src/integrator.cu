#include <vulcan/integrator.h>
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

VULCAN_GLOBAL
void IntegrateKernel(const int* indices, const HashEntry* hash_entries,
    float voxel_length, float block_length, float truncation_length,
    const float* depths, const Vector3f* colors, int image_width,
    int image_height, float max_weight, const Projection projection,
    const Transform Tcw, Voxel* voxels)
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
    // get measurement from depth image
    const int image_index = int(uv[1]) * image_width + int(uv[0]);
    const float depth = depths[image_index];

    // ignore invalid depth values
    if (depth <= 0.1f || depth >= 5.0f) return; // TODO: expose parameters

    // compute signed distance
    const float distance = depth - Xcp[2];

    // check if within truncated segment
    if (distance > -truncation_length)
    {
      // compute voxel data index
      const int r = Block::resolution;
      const int r2 = Block::resolution * Block::resolution;
      const int block_index = entry.data * Block::voxel_count;
      const int voxel_index = block_index + z * r2 + y * r + x;

      // update voxel data
      Voxel voxel = voxels[voxel_index];
      const Vector3f curr_color = colors[image_index];
      const float prev_dist = voxel.weight * voxel.distance;
      const float curr_dist = min(1.0f, distance / truncation_length);
      const Vector3f prev_color = voxel.weight * voxel.GetColor();
      const float new_weight = voxel.weight + 1;
      voxel.weight = min(max_weight, new_weight);
      voxel.distance = (prev_dist + curr_dist) / new_weight;
      voxel.SetColor((prev_color + curr_color) / new_weight);
      voxels[voxel_index] = voxel;
    }
  }
}

Integrator::Integrator(std::shared_ptr<Volume> volume) :
  volume_(volume),
  max_weight_(16)
{
}

std::shared_ptr<Volume> Integrator::GetVolume() const
{
  return volume_;
}

float Integrator::GetMaxWeight() const
{
  return max_weight_;
}

void Integrator::SetMaxWeight(float weight)
{
  VULCAN_DEBUG(weight > 0);
  max_weight_ = weight;
}

void Integrator::Integrate(const Frame& frame)
{
  // TODO: handle color and depth images that aren't co-located
  // for all visible voxels, project onto depth image and integrate
  // for all visible voxels, project onto color image and integrate
  // should check to see if current voxel is withing truncation band though
  // simply ignore occlusions

  const Buffer<int>& index_buffer = volume_->GetVisibleBlocks();
  const Buffer<HashEntry>& entry_buffer = volume_->GetHashEntries();
  Buffer<Voxel>& voxel_buffer = volume_->GetVoxels();

  const int* indices = index_buffer.GetData();
  const HashEntry* entries = entry_buffer.GetData();
  const float voxel_length = volume_->GetVoxelLength();
  const float block_length = Block::resolution * voxel_length;
  const float truncation_length = volume_->GetTruncationLength();
  const float* depths = frame.depth_image->GetData();
  const Vector3f* colors = frame.color_image->GetData();
  const int image_width = frame.depth_image->GetWidth();
  const int image_height = frame.depth_image->GetHeight();
  const Projection& projection = frame.projection;
  const Transform& Tcw = frame.Tcw;
  Voxel* voxels = voxel_buffer.GetData();

  const int resolution = Block::resolution;
  const size_t blocks = index_buffer.GetSize();
  const dim3 threads(resolution, resolution, resolution);

  CUDA_LAUNCH(IntegrateKernel, blocks, threads, 0, 0, indices, entries,
      voxel_length, block_length, truncation_length, depths, colors,
      image_width, image_height, max_weight_, projection, Tcw, voxels);
}

} // namespace vulcan