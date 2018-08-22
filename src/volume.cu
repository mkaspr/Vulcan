#include <vulcan/volume.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <vulcan/block.h>
#include <vulcan/exception.h>
#include <vulcan/frame.h>
#include <vulcan/hash.h>
#include <vulcan/image.h>
#include <vulcan/util.cuh>
#include <vulcan/voxel.h>

namespace vulcan
{

VULCAN_DEVICE int buffer_size;

VULCAN_DEVICE int voxel_pointer;

VULCAN_DEVICE int excess_pointer;

template <int BLOCK_SIZE>
VULCAN_GLOBAL
void UpdateBlockVisibilityKernel(const HashEntry* hash_entries,
    Visibility* block_visibility, int* visible_blocks, float block_length,
    int image_width, int image_height, const Projection projection,
    const Transform Tcw, int count)
{
  bool visible = false;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  // check if valid kernel thread
  if (index < count)
  {
    const Visibility visibility = block_visibility[index];
    visible = (visibility == VISIBILITY_TRUE);

    // check if visibility check needed
    if (visibility == VISIBILITY_UNKNOWN)
    {
      // get current block origin
      const Block block = hash_entries[index].block;
      const Vector3s& origin = block.GetOrigin();
      Vector4f Xwp;
      Xwp[3] = 1;

      // for each corner of block
      for (int i = 0; i < 8; ++i)
      {
        // compute position of corner in world frame
        Xwp[0] = block_length * (origin[0] + ((i & 0b001) >> 0));
        Xwp[1] = block_length * (origin[1] + ((i & 0b010) >> 1));
        Xwp[2] = block_length * (origin[2] + ((i & 0b100) >> 2));

        // converte point to camera frame
        const Vector4f Xcp = Tcw * Xwp;

        // check if behind camera
        if (Xcp[2] < 0) continue;

        // project point to image plane
        const Vector2f uv = projection.Project(Vector3f(Xcp));

        // check if point inside field of view
        if (uv[0] >= 0 && uv[0] <= image_width &&
            uv[1] >= 0 && uv[1] <= image_height)
        {
          visible = true;
          break;
        }
      }

      // update unknown visibility status only if not visible
      if (!visible) block_visibility[index] = VISIBILITY_FALSE;
    }
  }

  // compute index in output buffer
  const int offset = PrefixSum<BLOCK_SIZE>(visible, threadIdx.x, buffer_size);

  // write valid entries to output buffer
  if (offset >= 0) visible_blocks[offset] = index;
}

VULCAN_GLOBAL
void CreateAllocationRequestsKernel(const HashEntry* hash_entries,
    Visibility* block_visibility, AllocationType* allocation_types,
    Block* allocation_blocks, const float* depths, int image_width,
    int image_height, Projection projection, Transform Twc,
    int block_count, float block_length, float truncation_length)
{
  // compute pixel indices
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  // check if pixel in image
  if (x < image_width && y < image_height)
  {
    // convert pixel indices to pixel centers
    const Vector2f uv(x + 0.5f, y + 0.5f);

    // create ray direction from camera projection
    Vector3f direction = projection.Unproject(uv);

    // transform direction from camera frame to world frame
    direction = Vector3f(Twc * Vector4f(direction, 0));

    // get ray origin from camera center in world frame
    const Vector3f origin = Twc.GetTranslation();

    // get depth value from depth image at pixel index
    const float depth = depths[y * image_width + x];

    // ignore invalid depth values
    if (depth <= 0.1f || depth >= 5.0f) return; // TODO: expose parameters

    // compute inferred 3D point in world frame frame depth
    // NOTE: assume depth values are distance from image plane along Z-axis
    // and that the direction from unproject is scaled such that dir.z = 1
    const Vector3f Xwp = origin + depth * direction;

    // normalize direction to magnitude of one
    direction.Normalize();

    // compute start and end points on line segment, given truncation length
    const Vector3f begin = Xwp - truncation_length * direction;
    const Vector3f end = Xwp + truncation_length * direction;

    // compute sign direction for each axis
    const int step_x = (direction[0] < 0) ? -1 : 1;
    const int step_y = (direction[1] < 0) ? -1 : 1;
    const int step_z = (direction[2] < 0) ? -1 : 1;

    // compute starting block index of segment (inclusive)
    const float inv_block_length = 1.0f / block_length;
    int bx = floorf(begin[0] * inv_block_length);
    int by = floorf(begin[1] * inv_block_length);
    int bz = floorf(begin[2] * inv_block_length);

    // compute ending block index of segment (inclusive)
    const int ex = floorf(end[0] * inv_block_length);
    const int ey = floorf(end[1] * inv_block_length);
    const int ez = floorf(end[2] * inv_block_length);

    // compute next blox boundary for each axis
    const float ox = block_length * (bx + max(0, step_x)) - begin[0];
    const float oy = block_length * (by + max(0, step_y)) - begin[1];
    const float oz = block_length * (bz + max(0, step_z)) - begin[2];

    // compute distance from start to next block along each axis
    float tmax_x = ox / direction[0];
    float tmax_y = oy / direction[1];
    float tmax_z = oz / direction[2];

    // TODO: handle nans better
    if (direction[0] == 0) tmax_x = 1E20;
    if (direction[1] == 0) tmax_y = 1E20;
    if (direction[2] == 0) tmax_z = 1E20;

    // compute length along the ray to reach tmax for each axis
    const float tdelta_x = (step_x * block_length) / direction[0];
    const float tdelta_y = (step_y * block_length) / direction[1];
    const float tdelta_z = (step_z * block_length) / direction[2];

    // define hash function constants
    const uint32_t P1 = 73856093;
    const uint32_t P2 = 19349669;
    const uint32_t P3 = 83492791;
    const uint32_t K  = block_count;

    // loop until beyond segment
    do
    {
      // create block from current indices
      const Block block(bx, by, bz);

      // compute hash code for current block
      const uint32_t hash_code = ((bx * P1) ^ (by * P2) ^ (bz * P3)) % K;

      // get hash entry for hash code
      HashEntry entry = hash_entries[hash_code];

      // check if match block found
      if (entry.block == block)
      {
        // block is already allocated
        // just need to mark block as visible
        block_visibility[hash_code] = VISIBILITY_TRUE;
      }
      // check if main entry is empty
      else if (!entry.IsAllocated())
      {
        // hash code has not yet been used
        // require block allocation in main entry list
        // also mark entry as visible as we already know its final location
        block_visibility[hash_code] = VISIBILITY_TRUE;
        allocation_types[hash_code] = ALLOC_TYPE_MAIN;
        allocation_blocks[hash_code] = block;
      }
      // entry in use, but know by the block in question
      else
      {
        bool found = false;
        uint32_t index = hash_code;

        // search until the end of the linked-list
        while (entry.HasNext())
        {
          // get next entry in list
          index = entry.next;
          entry = hash_entries[index];

          // check if entry matches our block
          if (entry.block == block)
          {
            // block was already allocated
            // we just need to mark is as visible
            // using the discovered index in the hash table
            block_visibility[index] = VISIBILITY_TRUE;
            found = true;
            break;
          }
        }

        // check if we failed to find block in linked list
        if (!found)
        {
          // block has not been allocated
          // and the hash code is already in use by another block
          // we need to mark the block for "excess" allocation
          // we don't yet know its final position
          // so we can mark the block as visible just yet
          // this will happen in "HandleAllocationRequests" kernel
          allocation_types[hash_code] = ALLOC_TYPE_EXCESS;
          allocation_blocks[hash_code] = block;
        }
      }

      // check if X-boundary will be reached before Y-boundary
      if (tmax_x < tmax_y)
      {
        // check if X-boundary will be reached before Z-boundary
        if (tmax_x < tmax_z)
        {
          // increment X block index
          bx += step_x;

          // check if we have reached the end of the segment
          if (bx == ex + step_x) break;

          // increment X travesal tally
          tmax_x += tdelta_x;
        }
        // Z-boundary will be reached first
        else
        {
          // increment Z block index
          bz += step_z;

          // check if we have reached the end of the segment
          if (bz == ez + step_z) break;

          // increment Z travesal tally
          tmax_z += tdelta_z;
        }
      }
      else
      {
        // check if Y-boundary will be reached before Z-boundary
        if (tmax_y < tmax_z)
        {
          // increment Y block index
          by += step_y;

          // check if we have reached the end of the segment
          if (by == ey + step_y) break;

          // increment Y travesal tally
          tmax_y += tdelta_y;
        }
        // Z-boundary will be reached first
        else
        {
          // increment Z block index
          bz += step_z;

          // check if we have reached the end of the segment
          if (bz == ez + step_z) break;

          // increment Z travesal tally
          tmax_z += tdelta_z;
        }
      }
    }
    // loop until end of segment reached
    // this will trigger an explicit break in the body of the loop
    while (true);
  }
}

VULCAN_GLOBAL
void HandleAllocationRequestsKernel(AllocationType* allocation_types,
    const Block* allocation_blocks, HashEntry* hash_entries,
    Visibility* block_visibility, const int* free_voxel_blocks,
    int max_count, int count)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  // check if valid kernel thread
  if (index < count)
  {
    const AllocationType type = allocation_types[index];

    // check if allocation requested at index
    if (type != ALLOC_TYPE_NONE)
    {
      HashEntry entry;
      int entry_index = index;
      entry.block = allocation_blocks[index];

      // check if excess entry allocation
      if (type == ALLOC_TYPE_EXCESS)
      {
        int other_index = index;
        HashEntry other = hash_entries[other_index];

        // search for last entry in link-list
        while (other.HasNext())
        {
          other_index = other.next;
          other = hash_entries[other_index];
        }

        // request new excess entry index
        entry_index = atomicAdd(&excess_pointer, 1);
        VULCAN_DEBUG_MSG(entry_index < max_count, "excess memory exhausted");

        // check if valid excess index returned
        if (entry_index < max_count)
        {
          // link with parent entry
          hash_entries[other_index].next = entry_index;

          // mark new block as visible
          block_visibility[entry_index] = VISIBILITY_TRUE;
        }
      }

      // request new voxel entry index
      const int voxel_index = atomicSub(&voxel_pointer, 1);
      VULCAN_DEBUG_MSG(voxel_index >= 0, "voxel memory exhausted");

      // check if both entry and voxel indices are valid
      if (entry_index < max_count && voxel_index >= 0)
      {
        entry.data = free_voxel_blocks[voxel_index];

        // assign new entry to hash table
        hash_entries[entry_index] = entry;
      }

      // mark allocation request as handled
      allocation_types[index] = ALLOC_TYPE_NONE;
    }
  }
}

Volume::Volume(int main_block_count, int excess_block_count) :
  max_block_count_(main_block_count + excess_block_count),
  main_block_count_(main_block_count),
  excess_block_count_(excess_block_count),
  truncation_length_(0.02),
  voxel_length_(0.008),
  empty_(true)
{
  Initialize();
}

int Volume::GetMainBlockCount() const
{
  return main_block_count_;
}

int Volume::GetExcessBlockCount() const
{
  return excess_block_count_;
}

float Volume::GetVoxelLength() const
{
  return voxel_length_;
}

void Volume::SetVoxelLength(float length)
{
  VULCAN_DEBUG(length > 0);
  voxel_length_ = length;
}

float Volume::GetTruncationLength() const
{
  return truncation_length_;
}

void Volume::SetTruncationLength(float length)
{
  VULCAN_DEBUG(length > 0);
  truncation_length_ = length;
}

void Volume::SetView(const Frame& frame)
{
  ResetBlockVisibility();
  CreateAllocationRequests(frame);
  HandleAllocationRequests();
  UpdateBlockVisibility(frame);
  empty_ = false;
}

const Buffer<HashEntry>& Volume::GetHashEntries() const
{
  return hash_entries_;
}

const Buffer<int>& Volume::GetAllocatedBlocks() const
{
  VULCAN_THROW("not implemented");
  VULCAN_DEVICE_RETURN(visible_blocks_);
}

const Buffer<int>& Volume::GetVisibleBlocks() const
{
  return visible_blocks_;
}

const Buffer<Voxel>& Volume::GetVoxels() const
{
  return voxels_;
}

Buffer<Voxel>& Volume::GetVoxels()
{
  return voxels_;
}

void Volume::ResetBlockVisibility()
{
  const int count = max_block_count_;
  thrust::device_ptr<Visibility> data(block_visibility_.GetData());
  thrust::replace(data, data + count, VISIBILITY_TRUE, VISIBILITY_UNKNOWN);
  CUDA_DEBUG_LAST();
}

void Volume::UpdateBlockVisibility(const Frame& frame)
{
  const HashEntry* hash_entries = hash_entries_.GetData();
  Visibility* block_visibility = block_visibility_.GetData();
  int* visible_blocks = visible_blocks_.GetData();
  const Projection& projection = frame.projection;
  const Transform& Tcw = frame.Tcw;
  const int image_width = frame.depth_image->GetWidth();
  const int image_height = frame.depth_image->GetHeight();
  const float block_length = Block::resolution * voxel_length_;

  const size_t threads = 512;
  const size_t total = max_block_count_;
  const size_t blocks = GetKernelBlocks(total, threads);

  ResetBufferSize();

  CUDA_LAUNCH(UpdateBlockVisibilityKernel<512>, blocks, threads, 0, 0,
      hash_entries, block_visibility, visible_blocks, block_length,
      image_width, image_height, projection, Tcw, max_block_count_);

  visible_blocks_.Resize(GetBufferSize());
}

void Volume::CreateAllocationRequests(const Frame& frame)
{
  const HashEntry* hash_entries = hash_entries_.GetData();
  Visibility* block_visibility = block_visibility_.GetData();
  AllocationType* allocation_types = allocation_types_.GetData();
  Block* allocation_blocks = allocation_blocks_.GetData();
  const float* depth = frame.depth_image->GetData();
  const int width = frame.depth_image->GetWidth();
  const int height = frame.depth_image->GetHeight();
  const Projection& projection = frame.projection;
  const Transform Twc = frame.Tcw.Inverse();
  const float block_length = Block::resolution * voxel_length_;

  const dim3 threads(16, 16);
  const dim3 total(width, height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(CreateAllocationRequestsKernel, blocks, threads, 0, 0,
    hash_entries, block_visibility, allocation_types,allocation_blocks,
    depth, width, height, projection, Twc, main_block_count_,
    block_length, truncation_length_);
}

void Volume::HandleAllocationRequests()
{
  AllocationType* allocation_types = allocation_types_.GetData();
  const Block* allocation_blocks = allocation_blocks_.GetData();
  const int* free_voxel_blocks = free_voxel_blocks_.GetData();
  Visibility* block_visibility = block_visibility_.GetData();
  HashEntry* hash_entries = hash_entries_.GetData();

  const size_t threads = 512;
  const size_t total = main_block_count_;
  const size_t blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(HandleAllocationRequestsKernel, blocks, threads, 0, 0,
      allocation_types, allocation_blocks, hash_entries, block_visibility,
      free_voxel_blocks, max_block_count_, main_block_count_);
}

int Volume::GetBufferSize() const
{
  int result;
  const size_t bytes = sizeof(int);
  CUDA_ASSERT(cudaMemcpyFromSymbol(&result, buffer_size, bytes));
  return result;
}

void Volume::ResetBufferSize() const
{
  const int result = 0;
  const size_t bytes = sizeof(int);
  CUDA_ASSERT(cudaMemcpyToSymbol(buffer_size, &result, bytes));
}

void Volume::Initialize()
{
  CreateVoxelBuffer();
  CreateHashEntryBuffer();
  CreateHashEntryPointer();
  CreateFreeVoxelBlockBuffer();
  CreateFreeVoxelBlockPointer();
  CreateAllocationTypeBuffer();
  CreateAllocationBlockBuffer();
  CreateBlockVisibilityBuffer();
  CreateVisibleBlockBuffer();
}

void Volume::CreateVoxelBuffer()
{
  voxels_.Resize(max_block_count_ * Block::voxel_count);
  thrust::device_ptr<Voxel> data(voxels_.GetData());
  thrust::fill(data, data + voxels_.GetSize(), Voxel::Empty());
  CUDA_DEBUG_LAST();
}

void Volume::CreateHashEntryBuffer()
{
  hash_entries_.Resize(max_block_count_);
  thrust::device_ptr<HashEntry> data(hash_entries_.GetData());
  thrust::fill(data, data + hash_entries_.GetSize(), HashEntry());
  CUDA_DEBUG_LAST();
}

void Volume::CreateHashEntryPointer()
{
  const size_t bytes = sizeof(int);
  const int value = main_block_count_;
  CUDA_DEBUG(cudaMemcpyToSymbol(excess_pointer, &value, bytes));
}

void Volume::CreateFreeVoxelBlockBuffer()
{
  free_voxel_blocks_.Resize(max_block_count_);
  thrust::device_ptr<int> data(free_voxel_blocks_.GetData());
  thrust::sequence(data, data + free_voxel_blocks_.GetSize());
  CUDA_DEBUG_LAST();
}

void Volume::CreateFreeVoxelBlockPointer()
{
  const size_t bytes = sizeof(int);
  const int value = free_voxel_blocks_.GetSize() - 1;
  CUDA_DEBUG(cudaMemcpyToSymbol(voxel_pointer, &value, bytes));
}

void Volume::CreateAllocationTypeBuffer()
{
  allocation_types_.Resize(main_block_count_);
  thrust::device_ptr<AllocationType> data(allocation_types_.GetData());
  thrust::fill(data, data + allocation_types_.GetSize(), ALLOC_TYPE_NONE);
  CUDA_DEBUG_LAST();
}

void Volume::CreateAllocationBlockBuffer()
{
  allocation_blocks_.Resize(main_block_count_);
}

void Volume::CreateBlockVisibilityBuffer()
{
  block_visibility_.Resize(max_block_count_);
  thrust::device_ptr<Visibility> data(block_visibility_.GetData());
  thrust::fill(data, data + block_visibility_.GetSize(), VISIBILITY_FALSE);
  CUDA_DEBUG_LAST();
}

void Volume::CreateVisibleBlockBuffer()
{
  visible_blocks_.Reserve(max_block_count_);
}

} // namespace vulcan