#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <opencv2/opencv.hpp>
#include <vulcan/device.h>
#include <vulcan/frame.h>
#include <vulcan/hash.h>
#include <vulcan/integrator.h>
#include <vulcan/tracer.cuh>
#include <vulcan/tracer.h>
#include <vulcan/volume.h>
#include <vulcan/voxel.h>

namespace vulcan
{
namespace testing
{

TEST(Tracer, ComputePatches)
{
  thrust::host_vector<int> indices;
  thrust::host_vector<HashEntry> entries;
  thrust::host_vector<Patch> expected_patches;

  float block_length = 0.008;
  const int image_width = 640;
  const int image_height = 480;
  const int bounds_width = 80;
  const int bounds_height = 60;

  const Transform Tcw =
      Transform::Translate(0.3f, -1.3f, 3.7f) *
      Transform::Rotate(0.7474f, 0.3438f, -0.3884f, 0.4152f);

  Projection projection;
  projection.SetFocalLength(346.723f, 353.914f);
  projection.SetCenterPoint(321.294f, 239.052f);

  std::vector<Vector3f> points;
  points.push_back(Vector3f(320.0f, 240.0f, 2.5f));
  points.push_back(Vector3f(120.0f, 340.0f, 1.5f));
  points.push_back(Vector3f(420.0f, 240.0f, 0.6f));
  points.push_back(Vector3f(-20.0f, -40.0f, 1.0f));
  points.push_back(Vector3f(720.0f, 580.0f, 1.0f));

  for (size_t i = 0; i < points.size(); ++i)
  {
    indices.push_back(i);
    const Vector3f& point = points[i];
    const Vector2f uv = Vector2f(point);
    const Vector3f Xcp = projection.Unproject(uv) * point[2];
    const Vector3f Xwp = Vector3f(Tcw.Inverse() * Vector4f(Xcp, 1.0f));

    HashEntry entry;
    entry.block[0] = Xwp[0] / block_length;
    entry.block[1] = Xwp[1] / block_length;
    entry.block[2] = Xwp[2] / block_length;
    entry.next = -1;
    entry.data = 0;

    entries.push_back(entry);

    Vector4f corner(0, 0, 0, 1);
    Vector2i bmin(INT_MAX, INT_MAX);
    Vector2i bmax(INT_MIN, INT_MIN);
    Vector2f drng(+FLT_MAX, -FLT_MAX);

    for (int z = 0; z <= 1; ++z)
    {
      corner[2] = block_length * (z + entry.block[2]);

      for (int y = 0; y <= 1; ++y)
      {
        corner[1] = block_length * (y + entry.block[1]);

        for (int x = 0; x <= 1; ++x)
        {
          corner[0] = block_length * (x + entry.block[0]);
          const Vector3f Xcp = Vector3f(Tcw * corner);
          const Vector2f uv = projection.Project(Xcp);

          const float u = bounds_width * (uv[0] / image_width);
          const float v = bounds_height * (uv[1] / image_height);

          bmin[0] = clamp(min((int)floorf(u), bmin[0]), 0, bounds_width - 1);
          bmin[1] = clamp(min((int)floorf(v), bmin[1]), 0, bounds_height - 1);

          bmax[0] = clamp(max((int)ceilf(u), bmax[0]), 0, bounds_width - 1);
          bmax[1] = clamp(max((int)ceilf(v), bmax[1]), 0, bounds_height - 1);

          drng[0] = min(Xcp[2], drng[0]);
          drng[1] = max(Xcp[2], drng[1]);
        }
      }
    }

    const Vector2i diff = bmax - bmin;
    const int gx = (diff[0] + Patch::max_size - 1) / Patch::max_size;
    const int gy = (diff[1] + Patch::max_size - 1) / Patch::max_size;

    Patch patch;
    patch.bounds = drng;

    for (int j = 0; j < gy; ++j)
    {
      patch.origin[1] = bmin[1] + Patch::max_size * j;
      patch.size[1] = min(bmax[1] - patch.origin[1] + 1, Patch::max_size);

      for (int k = 0; k < gx; ++k)
      {
        patch.origin[0] = bmin[0] + Patch::max_size * k;
        patch.size[0] = min(bmax[0] - patch.origin[0] + 1, Patch::max_size);
        expected_patches.push_back(patch);
      }
    }
  }

  const int block_count = indices.size();
  thrust::device_vector<int> d_indices(indices);
  thrust::device_vector<HashEntry> d_entries(entries);
  thrust::device_vector<Patch> d_found_patches(10 * expected_patches.size());
  thrust::device_vector<int> d_found_count(1);

  const int* p_indices = d_indices.data().get();
  const HashEntry* p_entries = d_entries.data().get();
  Patch* p_found_patches = d_found_patches.data().get();
  int* p_found_count = d_found_count.data().get();

  vulcan::ComputePatches(p_indices, p_entries, Tcw, projection, block_length,
      block_count, image_width, image_height, bounds_width, bounds_height,
      p_found_patches, p_found_count);

  thrust::host_vector<Patch> found_patches(d_found_patches);
  ASSERT_EQ(expected_patches.size(), d_found_count[0]);

  for (size_t i = 0; i < expected_patches.size(); ++i)
  {
    bool matched = false;
    const Patch& found = found_patches[i];

    for (size_t j = 0; j < expected_patches.size(); ++j)
    {
      Patch& expected = expected_patches[j];

      if (fabsf(expected.bounds[0] - found.bounds[0]) < 1E-5f &&
          fabsf(expected.bounds[1] - found.bounds[1]) < 1E-5f &&
          expected.origin[0] == found.origin[0] &&
          expected.origin[1] == found.origin[1] &&
          expected.size[0] == found.size[0] &&
          expected.size[1] == found.size[1])
      {
        matched = true;
        expected.bounds[0] = std::numeric_limits<float>::quiet_NaN();
        break;
      }
    }

    ASSERT_TRUE(matched);
  }
}

TEST(Tracer, ComputeBounds)
{
  const int bounds_width = 80;
  const int bounds_height = 60;
  const int bounds_count = bounds_width * bounds_height;
  const Vector2f init(+FLT_MAX, -FLT_MAX);

  thrust::host_vector<Patch> patches;
  thrust::host_vector<Vector2f> expected(bounds_count);
  thrust::fill(expected.begin(), expected.end(), init);

  Patch patch;

  patch.origin = Vector2s(23, 46);
  patch.size = Vector2s(5, 2);
  patch.bounds = Vector2f(1.237, 1.523);
  patches.push_back(patch);

  patch.origin = Vector2s(3, 9);
  patch.size = Vector2s(1, 1);
  patch.bounds = Vector2f(2.021, 3.214);
  patches.push_back(patch);

  patch.origin = Vector2s(20, 43);
  patch.size = Vector2s(5, 8);
  patch.bounds = Vector2f(0.856, 1.014);
  patches.push_back(patch);

  patch.origin = Vector2s(0, 0);
  patch.size = Vector2s(2, 2);
  patch.bounds = Vector2f(1.256, 2.114);
  patches.push_back(patch);

  patch.origin = Vector2s(79, 59);
  patch.size = Vector2s(1, 1);
  patch.bounds = Vector2f(0.256, 1.314);
  patches.push_back(patch);

  patch.origin = Vector2s(3, 9);
  patch.size = Vector2s(3, 3);
  patch.bounds = Vector2f(0.256, 1.314);
  patches.push_back(patch);

  for (const Patch& patch : patches)
  {
    for (int i = 0; i < patch.size[1]; ++i)
    {
      const int y = patch.origin[1] + i;

      for (int j = 0; j < patch.size[0]; ++j)
      {
        const int x = patch.origin[0] + j;
        const int pixel = y * bounds_width + x;
        expected[pixel][0] = min(patch.bounds[0], expected[pixel][0]);
        expected[pixel][1] = max(patch.bounds[1], expected[pixel][1]);
      }
    }
  }

  const int patch_count = patches.size();
  thrust::device_vector<Patch> d_patches(patches);
  thrust::device_vector<Vector2f> d_found(expected.size());

  const Patch* p_patches = d_patches.data().get();
  Vector2f* p_found = d_found.data().get();

  vulcan::ResetBoundsBuffer(p_found, d_found.size());
  vulcan::ComputeBounds(p_patches, p_found, bounds_width, patch_count);

  thrust::host_vector<Vector2f> found(d_found);

  for (size_t i = 0; i < expected.size(); ++i)
  {
    ASSERT_FLOAT_EQ(expected[i][0], found[i][0]);
    ASSERT_FLOAT_EQ(expected[i][1], found[i][1]);
  }
}

TEST(Tracer, ComputePoints)
{
  const int image_width = 640;
  const int image_height = 480;
  const int bounds_width = 80;
  const int bounds_height = 60;
  const float trunc_length = 0.02;
  const float voxel_length = 0.008;
  const float block_length = voxel_length * Block::resolution;

  std::shared_ptr<Volume> volume;
  volume = std::make_shared<Volume>(4096, 2048);
  volume->SetTruncationLength(trunc_length);
  volume->SetVoxelLength(voxel_length);

  Frame frame;

  // const Transform Tcw =
  //     Transform::Translate(0.3f, -1.3f, 3.7f) *
  //     Transform::Rotate(0.7474f, 0.3438f, -0.3884f, 0.4152f);

  const Transform Tcw =
      Transform::Rotate(0.7474f, 0.3438f, -0.3884f, 0.4152f);

  const Transform Twc = Tcw.Inverse();

  // Projection projection;
  // projection.SetFocalLength(546.723f, 553.914f);
  // projection.SetCenterPoint(321.294f, 239.052f);

  Projection projection;
  projection.SetFocalLength(547, 547);
  projection.SetCenterPoint(320, 240);

  const float init_depth = 1.5f;
  std::shared_ptr<Image> depth_image;
  depth_image = std::make_shared<Image>(image_width, image_height);
  thrust::host_vector<float> expected_depths(depth_image->GetTotal());
  thrust::fill(expected_depths.begin(), expected_depths.end(), init_depth);
  thrust::device_ptr<float> depth_ptr(depth_image->GetData());
  thrust::copy(expected_depths.begin(), expected_depths.end(), depth_ptr);

  const Vector3f init_color(0.1, 0.2, 0.3);
  std::shared_ptr<ColorImage> color_image;
  color_image = std::make_shared<ColorImage>(image_width, image_height);
  thrust::host_vector<Vector3f> expected_colors(color_image->GetTotal());
  thrust::fill(expected_colors.begin(), expected_colors.end(), init_color);
  thrust::device_ptr<Vector3f> color_ptr(color_image->GetData());
  thrust::copy(expected_colors.begin(), expected_colors.end(), color_ptr);

  frame.Tcw = Tcw;
  frame.projection = projection;
  frame.depth_image = depth_image;
  frame.color_image = color_image;

  size_t visible_count = 0;

  do
  {
    visible_count = volume->GetVisibleBlocks().GetSize();
    volume->SetView(frame);
  }
  while (visible_count != volume->GetVisibleBlocks().GetSize());

  Integrator integrator(volume);
  integrator.Integrate(frame);

  const int block_count = volume->GetMainBlockCount();
  thrust::device_vector<Patch> d_patches(8192);
  thrust::device_vector<Vector2f> d_bounds(bounds_width * bounds_height);
  thrust::device_vector<float> d_found_depths(depth_image->GetTotal());
  thrust::device_vector<Vector3f> d_found_colors(color_image->GetTotal());
  thrust::device_vector<int> d_patch_count(1);

  const int* p_indices = volume->GetVisibleBlocks().GetData();
  const HashEntry* p_entries = volume->GetHashEntries().GetData();
  const Voxel* p_voxels = volume->GetVoxels().GetData();
  Patch* p_patches = d_patches.data().get();
  Vector2f* p_bounds = d_bounds.data().get();
  float* p_found_depths = d_found_depths.data().get();
  Vector3f* p_found_colors = d_found_colors.data().get();
  int* p_patch_count = d_patch_count.data().get();

  d_patch_count[0] = 0;

  vulcan::ComputePatches(p_indices, p_entries, Tcw, projection, block_length,
      visible_count, image_width, image_height, bounds_width, bounds_height,
      p_patches, p_patch_count);

  const int patch_count = d_patch_count[0];

  vulcan::ResetBoundsBuffer(p_bounds, d_bounds.size());
  vulcan::ComputeBounds(p_patches, p_bounds, bounds_width, patch_count);

  vulcan::ComputePoints(p_entries, p_voxels, p_bounds, block_count,
      block_length, voxel_length, trunc_length, Twc, projection, p_found_depths,
      p_found_colors, image_width, image_height, bounds_width, bounds_height);

  thrust::host_vector<float> found_depths(d_found_depths);
  thrust::host_vector<Vector3f> found_colors(d_found_colors);

  {
    cv::Mat image(image_height, image_width, CV_32FC1, found_depths.data());
    image.convertTo(image, CV_16UC1, 10000);
    cv::imwrite("depth.png", image);
  }

  {
    cv::Mat image(image_height, image_width, CV_32FC3, found_colors.data());
    image.convertTo(image, CV_8UC3, 255);
    cv::cvtColor(image, image, CV_BGR2RGB);
    cv::imwrite("color.png", image);
  }

  for (size_t i = 0; i < expected_depths.size(); ++i)
  {
    const int x = i % image_width;
    const int y = i / image_width;

    if (x <= 2 || x >= image_width  - 2 ||
        y <= 2 || y >= image_height - 2)
    {
      continue;
    }

    const float found = found_depths[i];
    const float expected = expected_depths[i];

    ASSERT_NEAR(expected, found, 0.01);
  }

  for (size_t i = 0; i < expected_colors.size(); ++i)
  {
    const int x = i % image_width;
    const int y = i / image_width;

    if (x <= 2 || x >= image_width  - 2 ||
        y <= 2 || y >= image_height - 2)
    {
      continue;
    }

    const Vector3f& found = found_colors[i];
    const Vector3f& expected = expected_colors[i];
    ASSERT_FLOAT_EQ(expected[0], found[0]);
    ASSERT_FLOAT_EQ(expected[1], found[1]);
    ASSERT_FLOAT_EQ(expected[2], found[2]);
  }
}

} // namespace testing

} // namespace vulcan