#pragma once

#include <cstdint>
#include <vulcan/buffer.h>
#include <vulcan/types.h>

namespace vulcan
{

class Block;
class Frame;
class HashEntry;
class Voxel;

class Volume
{
  public:

    Volume(int main_block_count, int excess_block_count);

    int GetMainBlockCount() const;

    int GetExcessBlockCount() const;

    float GetVoxelLength() const;

    void SetVoxelLength(float length);

    float GetTruncationLength() const;

    void SetTruncationLength(float length);

    void SetView(const Frame& frame);

    const Buffer<HashEntry>& GetHashEntries() const;

    const Buffer<int>& GetAllocatedBlocks() const;

    const Buffer<int>& GetVisibleBlocks() const;

    const Buffer<Voxel>& GetVoxels() const;

    Buffer<Voxel>& GetVoxels();

  protected:

    void ResetBlockVisibility();

    void UpdateBlockVisibility(const Frame& frame);

    void CreateAllocationRequests(const Frame& frame);

    void HandleAllocationRequests();

    int GetBufferSize() const;

    void ResetBufferSize() const;

  private:

    void Initialize();

    void CreateVoxelBuffer();

    void CreateHashEntryBuffer();

    void CreateHashEntryPointer();

    void CreateFreeVoxelBlockBuffer();

    void CreateFreeVoxelBlockPointer();

    void CreateAllocationTypeBuffer();

    void CreateAllocationBlockBuffer();

    void CreateBlockVisibilityBuffer();

    void CreateVisibleBlockBuffer();

  protected:

    Buffer<Voxel> voxels_;

    Buffer<HashEntry> hash_entries_;

    Buffer<int> free_voxel_blocks_;

    Buffer<AllocationType> allocation_types_;

    Buffer<Block> allocation_blocks_;

    Buffer<Visibility> block_visibility_;

    Buffer<int> visible_blocks_;

    int max_block_count_;

    int main_block_count_;

    int excess_block_count_;

    float truncation_length_;

    float voxel_length_;

    bool empty_;
};

} // namespace vulcan