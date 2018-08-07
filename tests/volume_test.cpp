#include <algorithm>
#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <vulcan/block.h>
#include <vulcan/buffer.h>
#include <vulcan/frame.h>
#include <vulcan/hash.h>
#include <vulcan/types.h>
#include <vulcan/volume.h>

#define MAIN_BLOCK_COUNT 1024
#define EXCESS_BLOCK_COUNT 512
#define MAX_BLOCK_COUNT (MAIN_BLOCK_COUNT + EXCESS_BLOCK_COUNT)

namespace vulcan
{
namespace testing
{
namespace
{

class Volume : public vulcan::Volume, public ::testing::Test
{
  public:

    Volume() :
      vulcan::Volume(MAIN_BLOCK_COUNT, EXCESS_BLOCK_COUNT)
    {
    }
};

} // namespace

TEST_F(Volume, Constructor)
{
  ASSERT_FLOAT_EQ(0.008, GetVoxelLength());
  ASSERT_FLOAT_EQ(0.02, GetTruncationLength());
  ASSERT_EQ(MAX_BLOCK_COUNT, max_block_count_);
  ASSERT_EQ(MAX_BLOCK_COUNT * Block::voxel_count, voxels_.GetCapacity());
  ASSERT_EQ(MAX_BLOCK_COUNT * Block::voxel_count, voxels_.GetSize());
  ASSERT_EQ(MAX_BLOCK_COUNT, hash_entries_.GetCapacity());
  ASSERT_EQ(MAX_BLOCK_COUNT, hash_entries_.GetSize());
  ASSERT_EQ(MAX_BLOCK_COUNT, free_voxel_blocks_.GetCapacity());
  ASSERT_EQ(MAX_BLOCK_COUNT, free_voxel_blocks_.GetSize());
  ASSERT_EQ(MAIN_BLOCK_COUNT, allocation_types_.GetCapacity());
  ASSERT_EQ(MAIN_BLOCK_COUNT, allocation_types_.GetSize());
  ASSERT_EQ(MAIN_BLOCK_COUNT, allocation_blocks_.GetCapacity());
  ASSERT_EQ(MAIN_BLOCK_COUNT, allocation_blocks_.GetSize());
  ASSERT_EQ(MAX_BLOCK_COUNT, block_visibility_.GetCapacity());
  ASSERT_EQ(MAX_BLOCK_COUNT, block_visibility_.GetSize());
  ASSERT_EQ(MAX_BLOCK_COUNT, visible_blocks_.GetCapacity());
  ASSERT_EQ(0, visible_blocks_.GetSize());
  ASSERT_TRUE(empty_);

  {
    thrust::device_ptr<const HashEntry> ptr(hash_entries_.GetData());
    thrust::host_vector<HashEntry> data(ptr, ptr + MAX_BLOCK_COUNT);
    HashEntry expected;

    for (const HashEntry& datum : data)
    {
      ASSERT_EQ(expected.block, datum.block);
      ASSERT_EQ(expected.data, datum.data);
      ASSERT_EQ(expected.next, datum.next);
    }
  }

  {
    thrust::device_ptr<const Visibility> ptr(block_visibility_.GetData());
    thrust::host_vector<Visibility> data(ptr, ptr + MAX_BLOCK_COUNT);

    for (Visibility datum : data)
    {
      ASSERT_EQ(VISIBILITY_FALSE, datum);
    }
  }

  {
    thrust::device_ptr<const AllocationType> ptr(allocation_types_.GetData());
    thrust::host_vector<AllocationType> data(ptr, ptr + MAIN_BLOCK_COUNT);

    for (AllocationType datum : data)
    {
      ASSERT_EQ(ALLOC_TYPE_NONE, datum);
    }
  }

  {
    thrust::device_ptr<const int> ptr(free_voxel_blocks_.GetData());
    thrust::host_vector<int> data(ptr, ptr + MAX_BLOCK_COUNT);

    for (size_t i = 0; i < data.size(); ++i)
    {
      ASSERT_EQ(i, data[i]);
    }
  }
}

TEST_F(Volume, ResetBlockVisibility)
{
  thrust::host_vector<Visibility> found(MAX_BLOCK_COUNT);
  thrust::host_vector<Visibility> expected(MAX_BLOCK_COUNT);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    expected[i] = (i % 7 == 0) ? VISIBILITY_TRUE : VISIBILITY_FALSE;
  }

  thrust::device_ptr<Visibility> ptr(block_visibility_.GetData());
  thrust::copy(expected.begin(), expected.end(), ptr);
  ResetBlockVisibility();
  thrust::copy(ptr, ptr + MAX_BLOCK_COUNT, found.begin());

  for (size_t i = 0; i < expected.size(); ++i)
  {
    const Visibility value = (expected[i] == VISIBILITY_TRUE) ?
          VISIBILITY_UNKNOWN : VISIBILITY_FALSE;

    ASSERT_EQ(value, found[i]);
  }
}

TEST_F(Volume, UpdateBlockVisibility)
{
  Frame frame;
  size_t found_count;
  size_t expected_count;
  std::vector<bool> expected(MAX_BLOCK_COUNT);
  thrust::device_ptr<int> ptr(visible_blocks_.GetData());

  frame.depth_image = std::make_shared<Image>(640, 480);
  frame.projection.SetFocalLength(320, 320);
  frame.projection.SetCenterPoint(320, 240);
  frame.Tcw = Transform::Translate(-10, 2, -30);

  {
    std::fill(expected.begin(), expected.end(), false);
    expected_count = 0;
    for (bool value : expected) expected_count += value;

    UpdateBlockVisibility(frame);
    found_count = visible_blocks_.GetSize();

    ASSERT_EQ(expected_count, found_count);

    thrust::host_vector<int> found(ptr, ptr + found_count);

    for (size_t i = 0; i < found.size(); ++i)
    {
      const int index = found[i];
      ASSERT_TRUE(expected[index]);
      expected[index] = false;
    }

    for (size_t i = 0; i < expected.size(); ++i)
    {
      ASSERT_FALSE(expected[i]);
    }
  }

  {
    std::fill(expected.begin(), expected.end(), false);
    expected[  7] = true;
    expected[ 32] = true;
    expected[123] = true;

    thrust::device_ptr<Visibility> block_vis(block_visibility_.GetData());
    block_vis[  7] = VISIBILITY_TRUE;
    block_vis[ 32] = VISIBILITY_TRUE;
    block_vis[123] = VISIBILITY_TRUE;

    expected_count = 0;
    for (bool value : expected) expected_count += value;

    UpdateBlockVisibility(frame);
    found_count = visible_blocks_.GetSize();

    ASSERT_EQ(expected_count, found_count);

    thrust::host_vector<int> found(ptr, ptr + found_count);

    for (size_t i = 0; i < found.size(); ++i)
    {
      const int index = found[i];
      ASSERT_TRUE(expected[index]);
      expected[index] = false;
    }

    for (size_t i = 0; i < expected.size(); ++i)
    {
      ASSERT_FALSE(expected[i]);
    }
  }

  {
    std::fill(expected.begin(), expected.end(), false);
    expected[  7] = true;
    expected[ 32] = true;
    expected[123] = true;
    expected[  3] = true;
    expected[315] = true;

    thrust::device_ptr<Visibility> block_vis(block_visibility_.GetData());
    block_vis[  7] = VISIBILITY_TRUE;
    block_vis[ 32] = VISIBILITY_TRUE;
    block_vis[123] = VISIBILITY_TRUE;
    block_vis[  3] = VISIBILITY_UNKNOWN;
    block_vis[ 17] = VISIBILITY_UNKNOWN;
    block_vis[315] = VISIBILITY_UNKNOWN;

    HashEntry entry;
    thrust::device_ptr<HashEntry> hash_entries(hash_entries_.GetData());

    entry.block = Block(10, -2, 33);
    hash_entries[  3] = entry;

    entry.block = Block(-10, -2, 28);
    hash_entries[ 17] = entry;

    entry.block = Block(11, -1, 53);
    hash_entries[315] = entry;

    expected_count = 0;
    for (bool value : expected) expected_count += value;

    UpdateBlockVisibility(frame);
    found_count = visible_blocks_.GetSize();

    ASSERT_EQ(expected_count, found_count);

    thrust::host_vector<int> found(ptr, ptr + found_count);

    for (size_t i = 0; i < found.size(); ++i)
    {
      const int index = found[i];
      ASSERT_TRUE(expected[index]);
      expected[index] = false;
    }

    for (size_t i = 0; i < expected.size(); ++i)
    {
      ASSERT_FALSE(expected[i]);
    }
  }
}

TEST_F(Volume, CreateAllocationRequests)
{
  Frame frame;
  frame.depth_image = std::make_shared<Image>(64, 48);
  frame.Tcw = Transform::Translate(-10.73, 2.11, -33.54);
  frame.projection.SetFocalLength(32, 32);
  frame.projection.SetCenterPoint(32, 24);

  const float trunc_length = 0.20;
  const float voxel_length = 0.02;
  const float block_length = Block::resolution * voxel_length;
  const float inv_block_length = 1 / block_length;
  SetTruncationLength(trunc_length);
  SetVoxelLength(voxel_length);

  const int w = frame.depth_image->GetWidth();
  const int h = frame.depth_image->GetHeight();
  thrust::host_vector<float> host_depth(w * h);

  const uint32_t P1 = 73856093;
  const uint32_t P2 = 19349669;
  const uint32_t P3 = 83492791;
  const uint32_t K  = MAIN_BLOCK_COUNT;

  thrust::host_vector<Block> exp_alloc_blocks(MAIN_BLOCK_COUNT);
  thrust::host_vector<AllocationType> exp_alloc_types(MAIN_BLOCK_COUNT);
  thrust::host_vector<Visibility> exp_visibility(MAX_BLOCK_COUNT);
  thrust::fill(exp_alloc_types.begin(), exp_alloc_types.end(), ALLOC_TYPE_NONE);
  thrust::fill(exp_visibility.begin(), exp_visibility.end(), VISIBILITY_FALSE);

  std::vector<bool> collisions(MAIN_BLOCK_COUNT);
  std::fill(collisions.begin(), collisions.end(), false);
  const Transform inv_transform = frame.Tcw.Inverse();

  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      const float d = 1 + 3 * (((x + y) % 100) / 99.0);

      host_depth[y * w + x] = d;
      const Vector2f uv(x + 0.5, y + 0.5);
      const Vector3f Xcp = d * frame.projection.Unproject(uv);
      const Vector3f Xwp = Vector3f(inv_transform * Vector4f(Xcp, 1));
      const Vector3f dir = (Xwp - inv_transform.GetTranslation()).Normalized();
      const Vector3f origin = Xwp - trunc_length * dir;

      float t = 0;
      Block prev_block;

      while (t < 2 * trunc_length + 1E-6f)
      {
        const Vector3f current = origin + t * dir;
        const int bx = floorf(current[0] * inv_block_length);
        const int by = floorf(current[1] * inv_block_length);
        const int bz = floorf(current[2] * inv_block_length);
        const Block curr_block(bx, by, bz);

        const uint32_t hash_code = ((bx * P1) ^ (by * P2) ^ (bz * P3)) % K;

        // const Block& block = curr_block;
        // if ((block[0] == 67 && block[1] == -18 && block[2] == 224) ||
        //     (x == 0 && y == 47))
        // {
        //   printf("(test) pixel: %d %d, t: %f, block: %d %d %d, hash: %u, K: %u\n",
        //       x, y, t, block[0], block[1], block[2], hash_code, K);
        // }

        if (exp_alloc_types[hash_code] == ALLOC_TYPE_MAIN &&
            exp_alloc_blocks[hash_code] != curr_block)
        {
          collisions[hash_code] = true;
        }

        exp_alloc_blocks[hash_code] = curr_block;
        exp_alloc_types[hash_code] = ALLOC_TYPE_MAIN;
        exp_visibility[hash_code] = VISIBILITY_TRUE;

        prev_block = curr_block;

        const int step_x = (dir[0] > 0) ? +1 : -1;
        const int step_y = (dir[1] > 0) ? +1 : -1;
        const int step_z = (dir[2] > 0) ? +1 : -1;

        const float delta_x = (block_length * (bx + step_x)) - current[0];
        const float delta_y = (block_length * (by + step_y)) - current[1];
        const float delta_z = (block_length * (bz + step_z)) - current[2];

        float rate_x = (dir[0] == 0) ? INFINITY : delta_x / dir[0];
        float rate_y = (dir[1] == 0) ? INFINITY : delta_y / dir[1];
        float rate_z = (dir[2] == 0) ? INFINITY : delta_z / dir[2];

        if (rate_x < 1E-8f) rate_x = 1E-6f;
        if (rate_y < 1E-8f) rate_y = 1E-6f;
        if (rate_z < 1E-8f) rate_z = 1E-6f;

        if (rate_x < rate_y)
        {
          if (rate_x < rate_z)
          {
            t += rate_x;
          }
          else
          {
            t += rate_z;
          }
        }
        else
        {
          if (rate_y < rate_z)
          {
            t += rate_y;
          }
          else
          {
            t += rate_z;
          }
        }
      }
    }
  }

  for (size_t i = 0; i < exp_alloc_types.size(); ++i)
  {
    if (exp_alloc_types[i] == ALLOC_TYPE_MAIN)
    {
      exp_visibility[i] = VISIBILITY_FALSE;
      exp_alloc_types[i] = ALLOC_TYPE_EXCESS;
      collisions[i] = true;

      HashEntry entry;
      entry.block = Block(-1, -1, -1);
      entry.data = 0;
      thrust::device_ptr<HashEntry> ptr(hash_entries_.GetData());
      ptr[i] = entry;
      break;
    }
  }

  thrust::device_ptr<float> dev_depth(frame.depth_image->GetData());
  thrust::copy(host_depth.begin(), host_depth.end(), dev_depth);
  CreateAllocationRequests(frame);

  {
    const size_t size = allocation_blocks_.GetSize();
    thrust::device_ptr<Block> ptr(allocation_blocks_.GetData());
    thrust::host_vector<Block> found(ptr, ptr + size);

    for (size_t i = 0; i < exp_alloc_blocks.size(); ++i)
    {
      if (!collisions[i] && exp_alloc_blocks[i] != found[i])
      {
        std::cout << "e: " << exp_alloc_blocks[i][0] << " " <<
                              exp_alloc_blocks[i][1] << " " <<
                              exp_alloc_blocks[i][2] << " " <<
                     "f: " << found[i][0] << " " <<
                              found[i][1] << " " <<
                              found[i][2] << std::endl;
      }

      // if (!collisions[i]) ASSERT_EQ(exp_alloc_blocks[i], found[i]);
    }

    // TODO: in case of collision, check to see if one of the blocks is correct
  }

  {
    const size_t size = allocation_types_.GetSize();
    thrust::device_ptr<AllocationType> ptr(allocation_types_.GetData());
    thrust::host_vector<AllocationType> found(ptr, ptr + size);

    for (size_t i = 0; i < exp_alloc_types.size(); ++i)
    {
      ASSERT_EQ(exp_alloc_types[i], found[i]);
    }
  }

  {
    const size_t size = block_visibility_.GetSize();
    thrust::device_ptr<Visibility> ptr(block_visibility_.GetData());
    thrust::host_vector<Visibility> found(ptr, ptr + size);

    for (size_t i = 0; i < exp_visibility.size(); ++i)
    {
      ASSERT_EQ(exp_visibility[i], found[i]);
    }
  }

  // TODO: hash force collisions
}

TEST_F(Volume, HandleAllocationRequests)
{
  thrust::host_vector<Block> h_blocks(MAIN_BLOCK_COUNT);
  thrust::host_vector<AllocationType> h_types(MAIN_BLOCK_COUNT);
  thrust::device_ptr<Block> d_blocks(allocation_blocks_.GetData());
  thrust::device_ptr<AllocationType> d_types(allocation_types_.GetData());
  thrust::fill(h_types.begin(), h_types.end(), ALLOC_TYPE_NONE);

  h_blocks[0] = Block(1, 2, 3);
  h_types[0] = ALLOC_TYPE_MAIN;

  h_blocks[323] = Block(7, 3, -1);
  h_types[323] = ALLOC_TYPE_MAIN;

  thrust::copy(h_blocks.begin(), h_blocks.end(), d_blocks);
  thrust::copy(h_types.begin(), h_types.end(), d_types);
  HandleAllocationRequests();

  {
    thrust::device_ptr<const AllocationType> ptr(allocation_types_.GetData());
    thrust::host_vector<AllocationType> data(ptr, ptr + MAIN_BLOCK_COUNT);

    for (AllocationType datum : data)
    {
      ASSERT_EQ(ALLOC_TYPE_NONE, datum);
    }
  }

  {
    thrust::device_ptr<const HashEntry> ptr(hash_entries_.GetData());
    thrust::host_vector<HashEntry> data(ptr, ptr + MAX_BLOCK_COUNT);
    HashEntry entry;
    Block block;

    int options[2];
    options[0] = MAX_BLOCK_COUNT - 1;
    options[1] = MAX_BLOCK_COUNT - 2;

    entry = data[0];
    block = entry.block;
    ASSERT_EQ(1, block.GetOrigin()[0]);
    ASSERT_EQ(2, block.GetOrigin()[1]);
    ASSERT_EQ(3, block.GetOrigin()[2]);
    ASSERT_FALSE(entry.HasNext());
    ASSERT_TRUE(entry.data == options[0] || entry.data == options[1]);

    entry = data[323];
    block = entry.block;
    ASSERT_EQ( 7, block.GetOrigin()[0]);
    ASSERT_EQ( 3, block.GetOrigin()[1]);
    ASSERT_EQ(-1, block.GetOrigin()[2]);
    ASSERT_FALSE(entry.HasNext());
    ASSERT_TRUE(entry.data == options[0] || entry.data == options[1]);
  }

  h_blocks[0] = Block(7, 3, 0);
  h_types[0] = ALLOC_TYPE_EXCESS;

  h_blocks[323] = Block(-9, 1, -2);
  h_types[323] = ALLOC_TYPE_EXCESS;

  thrust::copy(h_blocks.begin(), h_blocks.end(), d_blocks);
  thrust::copy(h_types.begin(), h_types.end(), d_types);
  HandleAllocationRequests();

  {
    thrust::device_ptr<const AllocationType> ptr(allocation_types_.GetData());
    thrust::host_vector<AllocationType> data(ptr, ptr + MAIN_BLOCK_COUNT);

    for (AllocationType datum : data)
    {
      ASSERT_EQ(ALLOC_TYPE_NONE, datum);
    }
  }

  {
    thrust::device_ptr<const HashEntry> ptr(hash_entries_.GetData());
    thrust::host_vector<HashEntry> data(ptr, ptr + MAX_BLOCK_COUNT);
    HashEntry entry;
    Block block;

    int opt_data[4];
    opt_data[0] = MAX_BLOCK_COUNT - 1;
    opt_data[1] = MAX_BLOCK_COUNT - 2;
    opt_data[2] = MAX_BLOCK_COUNT - 3;
    opt_data[3] = MAX_BLOCK_COUNT - 4;

    int opt_next[2];
    opt_next[0] = MAIN_BLOCK_COUNT + 0;
    opt_next[1] = MAIN_BLOCK_COUNT + 1;

    entry = data[0];
    block = entry.block;
    ASSERT_EQ(1, block.GetOrigin()[0]);
    ASSERT_EQ(2, block.GetOrigin()[1]);
    ASSERT_EQ(3, block.GetOrigin()[2]);
    ASSERT_TRUE(entry.next == opt_next[0] || entry.next == opt_next[1]);
    ASSERT_TRUE(entry.data == opt_data[0] || entry.data == opt_data[1]);

    entry = data[entry.next];
    block = entry.block;
    ASSERT_EQ(7, block.GetOrigin()[0]);
    ASSERT_EQ(3, block.GetOrigin()[1]);
    ASSERT_EQ(0, block.GetOrigin()[2]);
    ASSERT_FALSE(entry.HasNext());
    ASSERT_TRUE(entry.data == opt_data[2] || entry.data == opt_data[3]);

    entry = data[323];
    block = entry.block;
    ASSERT_EQ( 7, block.GetOrigin()[0]);
    ASSERT_EQ( 3, block.GetOrigin()[1]);
    ASSERT_EQ(-1, block.GetOrigin()[2]);
    ASSERT_TRUE(entry.next == opt_next[0] || entry.next == opt_next[1]);
    ASSERT_TRUE(entry.data == opt_data[0] || entry.data == opt_data[1]);

    entry = data[entry.next];
    block = entry.block;
    ASSERT_EQ(-9, block.GetOrigin()[0]);
    ASSERT_EQ( 1, block.GetOrigin()[1]);
    ASSERT_EQ(-2, block.GetOrigin()[2]);
    ASSERT_FALSE(entry.HasNext());
    ASSERT_TRUE(entry.data == opt_data[2] || entry.data == opt_data[3]);
  }
}

} // namespace testing

} // namespace vulcan