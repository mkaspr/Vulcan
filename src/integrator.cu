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
    const float* depth, int image_width, int image_height, float max_weight,
    const Projection projection, const Transform transform, Voxel* voxels)
{
  // TODO: voxel centers should be used (not corners)
  // using corners (and their consequent overlap) is needed for marching cubes
  // but marching cubes is not the focus of this pipeline
  // consequently, we should use voxele centers

  // get voxel indices
  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const int z = threadIdx.z;

  // get block hash table entry
  const int entry_index = indices[blockIdx.x];
  const HashEntry entry = hash_entries[entry_index];
  const Block& block = entry.block;

  // compute voxel point in world frame
  // TODO: FIX -0.5f HACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  const Vector3f block_offset = block_length * (-0.5f + Vector3f(block.GetOrigin()));
  const Vector3f voxel_offset = voxel_length * Vector3f(x, y, z);
  const Vector3f Xwp = block_offset + voxel_offset;

  // convert point to camera frame
  const Vector3f Xcp = Vector3f(transform * Vector4f(Xwp, 1));

  // project point to image plane
  const Vector2f uv = projection.Project(Xcp);

  // check if point inside image
  if (uv[0] >= 0 && uv[0] < image_width && uv[1] >= 0 && uv[1] < image_height)
  {
    // get measurement from depth image
    const float d = depth[int(uv[1]) * image_width + int(uv[0])];

    // ignore invalid depth values
    if (d <= 0.05f || d >= 5.0f) return; // TODO: expose parameters

    // compute signed distance
    const float distance = d - Xcp[2];

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
      const float prev_dist = voxel.weight * voxel.distance;
      const float curr_dist = min(1.0f, distance / truncation_length);
      const float new_weight = voxel.weight + 1;
      voxel.distance = (prev_dist + curr_dist) / new_weight;
      voxel.weight = min(max_weight, new_weight);
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
  const Buffer<int>& index_buffer = volume_->GetVisibleBlocks();
  const Buffer<HashEntry>& entry_buffer = volume_->GetHashEntries();
  Buffer<Voxel>& voxel_buffer = volume_->GetVoxels();

  const int* indices = index_buffer.GetData();
  const HashEntry* entries = entry_buffer.GetData();
  const float voxel_length = volume_->GetVoxelLength();
  const float block_length = (Block::resolution - 1) * voxel_length;
  const float truncation_length = volume_->GetTruncationLength();
  const float* depth = frame.depth_image->GetData();
  const int image_width = frame.depth_image->GetWidth();
  const int image_height = frame.depth_image->GetHeight();
  const Projection& projection = frame.projection;
  const Transform& transform = frame.transform;
  Voxel* voxels = voxel_buffer.GetData();

  const int resolution = Block::resolution;
  const size_t blocks = index_buffer.GetSize();
  const dim3 threads(resolution, resolution, resolution);

  CUDA_LAUNCH(IntegrateKernel, blocks, threads, 0, 0, indices, entries,
      voxel_length, block_length, truncation_length, depth, image_width,
      image_height, max_weight_, projection, transform, voxels);
}

} // namespace vulcan