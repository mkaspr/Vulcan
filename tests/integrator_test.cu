#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <vulcan/frame.h>
#include <vulcan/hash.h>
#include <vulcan/image.h>
#include <vulcan/integrator.h>
#include <vulcan/volume.h>
#include <vulcan/voxel.h>

namespace vulcan
{
namespace testing
{
namespace
{

class Volume : public vulcan::Volume
{
  public:

    Volume() :
      vulcan::Volume(512, 256)
    {
    }

    void GetVisibleBlocks(thrust::host_vector<int>& blocks) const
    {
      blocks.resize(visible_blocks_.GetSize());
      const int* raw_ptr = visible_blocks_.GetData();
      thrust::device_ptr<const int> device_ptr(raw_ptr);
      thrust::copy(device_ptr, device_ptr + blocks.size(), blocks.begin());
    }

    void GetHashEntries(thrust::host_vector<HashEntry>& entries) const
    {
      entries.resize(hash_entries_.GetSize());
      const HashEntry* raw_ptr = hash_entries_.GetData();
      thrust::device_ptr<const HashEntry> device_ptr(raw_ptr);
      thrust::copy(device_ptr, device_ptr + entries.size(), entries.begin());
    }

    void GetVoxels(thrust::host_vector<Voxel>& voxels) const
    {
      voxels.resize(voxels_.GetSize());
      const Voxel* raw_ptr = voxels_.GetData();
      thrust::device_ptr<const Voxel> device_ptr(raw_ptr);
      thrust::copy(device_ptr, device_ptr + voxels.size(), voxels.begin());
    }
};

} // namespace

TEST(Integrator, Constructor)
{
  std::shared_ptr<Volume> volume;
  volume = std::make_shared<Volume>();
  Integrator integrator(volume);
  ASSERT_EQ(volume, integrator.GetVolume());
  ASSERT_EQ(16, integrator.GetMaxWeight());
}

TEST(Integrator, MaxWeight)
{
  std::shared_ptr<Volume> volume;
  volume = std::make_shared<Volume>();
  Integrator integrator(volume);

  integrator.SetMaxWeight(10);
  ASSERT_EQ(10, integrator.GetMaxWeight());

  integrator.SetMaxWeight(2);
  ASSERT_EQ(2, integrator.GetMaxWeight());

#ifndef NDEBUG
  ASSERT_THROW(integrator.SetMaxWeight(0), Exception);
  ASSERT_THROW(integrator.SetMaxWeight(-1), Exception);
#endif
}

TEST(Integrator, Integrate)
{
  const int w = 160;
  const int h = 120;
  std::shared_ptr<Image> image;
  image = std::make_shared<Image>(w, h);
  thrust::device_ptr<float> ptr(image->GetData());
  const int count = image->GetTotal();
  thrust::fill(ptr, ptr + count, 1.5f);

  Frame frame;
  frame.transform = Transform::Translate(0, 0, 0);
  frame.projection.SetFocalLength(80, 80);
  frame.projection.SetCenterPoint(80, 60);
  frame.depth_image = image;

  const float trunc_length = 0.02;
  const float voxel_length = 0.008;
  const float block_length = (Block::resolution - 1) * voxel_length;

  std::shared_ptr<Volume> volume;
  volume = std::make_shared<Volume>();
  volume->SetTruncationLength(trunc_length);
  volume->SetVoxelLength(voxel_length);
  volume->SetView(frame);

  Integrator integrator(volume);
  integrator.SetMaxWeight(16);
  integrator.Integrate(frame);

  thrust::device_ptr<float> depth_ptr(image->GetData());
  thrust::host_vector<float> depth(depth_ptr, depth_ptr + image->GetTotal());

  thrust::host_vector<int> visible_blocks;
  volume->GetVisibleBlocks(visible_blocks);

  thrust::host_vector<HashEntry> hash_entries;
  volume->GetHashEntries(hash_entries);

  thrust::host_vector<Voxel> found;
  volume->GetVoxels(found);

  thrust::host_vector<Voxel> expected(found.size());
  thrust::fill(expected.begin(), expected.end(), Voxel::Empty());

  std::vector<bool> border_points(expected.size());
  std::fill(border_points.begin(), border_points.end(), false);

  for (size_t i = 0; i < visible_blocks.size(); ++i)
  {
    const int index = visible_blocks[i];
    const HashEntry& entry = hash_entries[index];
    const Vector3s block_index = entry.block.GetOrigin();
    const Vector3f block_offset = block_length * Vector3f(block_index);

    for (int z = 0; z < Block::resolution; ++z)
    {
      for (int y = 0; y < Block::resolution; ++y)
      {
        for (int x = 0; x < Block::resolution; ++x)
        {
          const Vector3f voxel_index = Vector3f(x, y, z);
          const Vector3f voxel_offset = voxel_length * voxel_index;
          const Vector3f Xwp = block_offset + voxel_offset;
          const Vector3f Xcp = Vector3f(frame.transform * Vector4f(Xwp, 1));
          const Vector2f uv = frame.projection.Project(Xcp);

          if (std::abs(uv[0]) < 1E-6f || std::abs(uv[0] - w) < 1E-6 ||
              std::abs(uv[1]) < 1E-6f || std::abs(uv[1] - h) < 1E-6)
          {
            const int r = Block::resolution;
            const int r2 = Block::resolution * Block::resolution;
            const int block_index = entry.data * Block::voxel_count;
            const int voxel_index = block_index + (z * r2 + y * r + x);
            border_points[voxel_index] = true;
          }

          if (uv[0] >= 0 && uv[0] < w && uv[1] >= 0 && uv[1] < h)
          {
            const Vector2i pixel(uv);
            const float d = depth[pixel[1] * w + pixel[0]];
            const float distance = d - Xcp[2];

            if (distance > -trunc_length)
            {
              const int r = Block::resolution;
              const int r2 = Block::resolution * Block::resolution;
              const int block_index = entry.data * Block::voxel_count;
              const int voxel_index = block_index + (z * r2 + y * r + x);
              Voxel& voxel = expected[voxel_index];

              const float prev_distance = voxel.weight * voxel.distance;
              const float curr_distance = min(1.0f, distance / trunc_length);
              const float new_weight = voxel.weight + 1;
              voxel.distance = (prev_distance + curr_distance) / new_weight;
              voxel.weight = new_weight;
            }
          }
        }
      }
    }
  }

  for (size_t i = 0; i < expected.size(); ++i)
  {
    if (!border_points[i] || (expected[i].weight > 0 && found[i].weight > 0))
    {
      ASSERT_NEAR(expected[i].distance, found[i].distance, 1E-6);
      ASSERT_NEAR(expected[i].weight, found[i].weight, 1E-6);
    }
  }

  integrator.Integrate(frame);
  volume->GetVoxels(found);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    if (!border_points[i] || (expected[i].weight > 0 && found[i].weight > 0))
    {
      ASSERT_NEAR(expected[i].distance, found[i].distance, 1E-6);
      ASSERT_NEAR(2 * expected[i].weight, found[i].weight, 1E-6);
    }
  }
}

} // namespace testing

} // namespace vulcan