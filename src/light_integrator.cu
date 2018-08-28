#include <vulcan/light_integrator.h>
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

template <int BLOCK_DIM, int KERNEL_SIZE>
VULCAN_GLOBAL
void ComputeFrameMaskKernel2(int width, int height, const float* depths,
    const Vector3f* colors, float depth_threshold, float* mask)
{
  // allocate shared memory
  const int buffer_dim = BLOCK_DIM + 2 * KERNEL_SIZE;
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
    float depth = 0;

    // get source image indices
    const int vx = (blockIdx.x * blockDim.x - 1) + (sindex % buffer_dim);
    const int vy = (blockIdx.y * blockDim.y - 1) + (sindex / buffer_dim);

    // check if within image bounds
    if (vx >= 0 && vx < width && vy >= 0 && vy < height)
    {
      // read value from global memory
      depth = depths[vy * width + vx];
    }

    // store value in shared memory
    buffer[sindex] = depth;

    // advance to next shared index
    sindex += block_size;
  }
  while (sindex < buffer_size);

  // wait for all threads to finish
  __syncthreads();

  // check if current thread within image bounds
  if (x < width && y < height)
  {
    // fetch pixel color from global memory
    const int index = y * width + x;
    const Vector3f value = colors[index];

    // check if pixel is improperly saturated
    if (value[0] < 0.02f || value[0] > 0.98f ||
        value[1] < 0.02f || value[1] > 0.98f ||
        value[2] < 0.02f || value[2] > 0.98f)
    {
      // store failure and exit
      mask[index] = false;
      return;
    }

    // initial extrema
    float dmin = +FLT_MAX;
    float dmax = -FLT_MAX;

    // get kernel center index
    const int cx = threadIdx.x + KERNEL_SIZE;
    const int cy = threadIdx.y + KERNEL_SIZE;

    // search over entire kernel patch
    for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; ++i)
    {
      for (int j = -KERNEL_SIZE; j <= KERNEL_SIZE; ++j)
      {
        // read depth from shared memory
        const float depth = buffer[(cy + i) * buffer_dim + (cx + j)];

        // update extrema
        dmin = fminf(depth, dmin);
        dmax = fmaxf(depth, dmax);
      }
    }

    // store result of depth discontinuity check
    mask[index] = (dmax - dmin <= depth_threshold);
  }
}

VULCAN_GLOBAL
void IntegrateKernel(const int* indices, const HashEntry* hash_entries,
    float voxel_length, const Light light, float block_length,
    float truncation_length, float min_depth, float max_depth,
    const float* frame_mask, const float* depths, const Vector3f* colors,
    const Vector3f* normals, int image_width, int image_height,
    float max_dist_weight, float max_color_weight, const Projection projection,
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
    if (depth < min_depth || depth > max_depth) return;

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

      // update voxel distance
      Voxel voxel = voxels[voxel_index];
      const float prev_dist = voxel.distance_weight * voxel.distance;
      const float curr_dist = min(1.0f, distance / truncation_length);
      const float dist_weight = voxel.distance_weight + 1;
      voxel.distance = (prev_dist + curr_dist) / dist_weight;
      voxel.distance_weight = min(max_dist_weight, dist_weight);

      // read image mask for pixel
      const float mask = frame_mask[image_index];

      // check if valid pixel for color integration
      if (mask > 0.5f)
      {
        // update voxel color
        Vector3f curr_color = colors[image_index];
        const Vector3f normal = normals[image_index];
        const float shading = light.GetShading(Xcp, normal);

        // ignore error-prone, low illumination scenarios
        if (shading > 0.05f) // TODO: expose parameter
        {
          curr_color /= shading;
          const float color_weight = voxel.color_weight + 1;
          const Vector3f prev_color = voxel.color_weight * voxel.GetColor();
          voxel.SetColor((prev_color + curr_color) / color_weight);
          voxel.color_weight = min(max_color_weight, color_weight);
        }
      }

      // assign updated voxel
      voxels[voxel_index] = voxel;
    }
  }
}

} // namespace

LightIntegrator::LightIntegrator(std::shared_ptr<Volume> volume) :
  Integrator(volume),
  depth_threshold_(0.2f)
{
}

const Light& LightIntegrator::GetLight() const
{
  return light_;
}

void LightIntegrator::SetLight(const Light& light)
{
  light_ = light;
}

void LightIntegrator::Integrate(const Frame& frame)
{
  // TODO: handle color and depth images that aren't co-located
  // for all visible voxels, project onto depth image and integrate
  // for all visible voxels, project onto color image and integrate
  // should check to see if current voxel is withing truncation band though
  // simply ignore occlusions

  ComputeFrameMask(frame);

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
  const Vector3f* normals = frame.normal_image->GetData();
  const int image_width = frame.depth_image->GetWidth();
  const int image_height = frame.depth_image->GetHeight();
  const Projection& projection = frame.projection;
  const Transform Tcw = frame.Twc.Inverse();
  const float* mask = frame_mask_.GetData();
  Voxel* voxels = voxel_buffer.GetData();

  const int resolution = Block::resolution;
  const size_t blocks = index_buffer.GetSize();
  const dim3 threads(resolution, resolution, resolution);

  CUDA_LAUNCH(IntegrateKernel, blocks, threads, 0, 0, indices, entries,
      voxel_length, light_, block_length, truncation_length,
      depth_range_[0], depth_range_[1], mask, depths, colors, normals,
      image_width, image_height, max_distance_weight_, max_color_weight_,
      projection, Tcw, voxels);
}

void LightIntegrator::ComputeFrameMask(const Frame& frame)
{
  const int w = frame.depth_image->GetWidth();
  const int h = frame.depth_image->GetHeight();

  const dim3 total(w, h);
  const dim3 threads(16, 16);
  const dim3 blocks = GetKernelBlocks(total, threads);

  frame_mask_.Resize(w, h);
  const float* depths = frame.depth_image->GetData();
  const Vector3f* colors = frame.color_image->GetData();
  float* mask = frame_mask_.GetData();

  CUDA_LAUNCH((ComputeFrameMaskKernel2<16, 3>), blocks, threads, 0, 0, w, h,
      depths, colors, depth_threshold_, mask);
}

} // namespace vulcan