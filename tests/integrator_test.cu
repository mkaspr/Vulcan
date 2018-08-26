#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <vulcan/color_integrator.h>
#include <vulcan/frame.h>
#include <vulcan/hash.h>
#include <vulcan/image.h>
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
  ColorIntegrator integrator(volume);
  ASSERT_EQ(volume, integrator.GetVolume());
  ASSERT_EQ(16, integrator.GetMaxDistanceWeight());
}

TEST(Integrator, MaxWeight)
{
  std::shared_ptr<Volume> volume;
  volume = std::make_shared<Volume>();
  ColorIntegrator integrator(volume);

  integrator.SetMaxDistanceWeight(10);
  ASSERT_EQ(10, integrator.GetMaxDistanceWeight());

  integrator.SetMaxDistanceWeight(2);
  ASSERT_EQ(2, integrator.GetMaxDistanceWeight());

#ifndef NDEBUG
  ASSERT_THROW(integrator.SetMaxDistanceWeight(0), Exception);
  ASSERT_THROW(integrator.SetMaxDistanceWeight(-1), Exception);
#endif
}

TEST(Integrator, Integrate)
{
  const int w = 160;
  const int h = 120;

  std::shared_ptr<Image> depth_image;
  depth_image = std::make_shared<Image>(w, h);
  thrust::device_ptr<float> ptr(depth_image->GetData());
  const int count = depth_image->GetTotal();
  thrust::fill(ptr, ptr + count, 1.5f);

  std::shared_ptr<ColorImage> color_image;
  color_image = std::make_shared<ColorImage>(w, h);
  thrust::device_ptr<Vector3f> cptr(color_image->GetData());
  const int color_count = color_image->GetTotal();
  thrust::fill(cptr, cptr + color_count, Vector3f(1, 2, 3));

  Frame frame;
  frame.Twc = Transform::Translate(0, 0, 0);
  frame.projection.SetFocalLength(80, 80);
  frame.projection.SetCenterPoint(80, 60);
  frame.depth_image = depth_image;
  frame.color_image = color_image;

  const float trunc_length = 0.02;
  const float voxel_length = 0.008;
  const float block_length = Block::resolution * voxel_length;

  std::shared_ptr<Volume> volume;
  volume = std::make_shared<Volume>();
  volume->SetTruncationLength(trunc_length);
  volume->SetVoxelLength(voxel_length);
  volume->SetView(frame);

  ColorIntegrator integrator(volume);
  integrator.SetMaxDistanceWeight(16);
  integrator.Integrate(frame);

  thrust::device_ptr<float> depth_ptr(depth_image->GetData());
  thrust::host_vector<float> depths(depth_ptr, depth_ptr + depth_image->GetTotal());

  thrust::device_ptr<Vector3f> color_ptr(color_image->GetData());
  thrust::host_vector<Vector3f> colors(color_ptr, color_ptr + color_image->GetTotal());

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
          const Vector3f voxel_offset = voxel_length * (voxel_index + 0.5f);
          const Vector3f Xwp = block_offset + voxel_offset;
          const Vector3f Xcp = Vector3f(frame.Twc.Inverse() * Vector4f(Xwp, 1));
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
            const int image_index = pixel[1] * w + pixel[0];
            const float depth = depths[image_index];
            const float distance = depth - Xcp[2];

            if (distance > -trunc_length)
            {
              const int r = Block::resolution;
              const int r2 = Block::resolution * Block::resolution;
              const int block_index = entry.data * Block::voxel_count;
              const int voxel_index = block_index + (z * r2 + y * r + x);
              Voxel& voxel = expected[voxel_index];

              const Vector3f curr_color = colors[image_index];
              const float prev_distance = voxel.distance_weight * voxel.distance;
              const float curr_distance = min(1.0f, distance / trunc_length);
              const Vector3f prev_color = voxel.distance_weight * voxel.GetColor();
              float new_weight = voxel.distance_weight + 1;
              voxel.distance = (prev_distance + curr_distance) / new_weight;
              voxel.SetColor((prev_color + curr_color) / new_weight);
              new_weight = min(integrator.GetMaxDistanceWeight(), new_weight);
              voxel.distance_weight = new_weight;
            }
          }
        }
      }
    }
  }

  for (size_t i = 0; i < expected.size(); ++i)
  {
    if (!border_points[i] || (expected[i].distance_weight > 0 && found[i].distance_weight > 0))
    {
      ASSERT_NEAR(expected[i].distance, found[i].distance, 1E-5);
      ASSERT_NEAR(expected[i].distance_weight, found[i].distance_weight, 1E-5);
    }
  }

  integrator.Integrate(frame);
  volume->GetVoxels(found);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    if (!border_points[i] || (expected[i].distance_weight > 0 && found[i].distance_weight > 0))
    {
      ASSERT_NEAR(expected[i].distance, found[i].distance, 1E-5);
      ASSERT_NEAR(2 * expected[i].distance_weight, found[i].distance_weight, 1E-5);
    }
  }
}

} // namespace testing

} // namespace vulcan