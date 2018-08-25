#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <vulcan/depth_tracker.h>
#include <vulcan/frame.h>

namespace vulcan
{
namespace testing
{

inline void CreateKeyframe(Frame& frame)
{
  int w = 640;
  int h = 320;

  Transform transform;
  frame.Tcw = transform;

  Projection projection;
  projection.SetFocalLength(547, 547);
  projection.SetCenterPoint(320, 240);
  frame.projection = projection;

  {
    std::shared_ptr<Image> depth_image;
    depth_image = std::make_shared<Image>(w, h);
    thrust::host_vector<float> depths(w * h);

    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        depths[y * w + x] = 1;
      }
    }

    thrust::device_ptr<float> ptr(depth_image->GetData());
    thrust::copy(depths.begin(), depths.end(), ptr);
    frame.depth_image = depth_image;
  }

  {
    std::shared_ptr<ColorImage> color_image;
    color_image = std::make_shared<ColorImage>(w, h);
    thrust::host_vector<Vector3f> colors(w * h);

    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        colors[y * w + x] = Vector3f(0.1, 0.2, 0.3);
      }
    }

    thrust::device_ptr<Vector3f> ptr(color_image->GetData());
    thrust::copy(colors.begin(), colors.end(), ptr);
    frame.color_image = color_image;
  }

  std::shared_ptr<ColorImage> normal_image;
  normal_image = std::make_shared<ColorImage>(w, h);
  frame.normal_image = normal_image;
  frame.ComputeNormals();
}

inline void CreateFrame(Frame& frame)
{
  int w = 640;
  int h = 320;

  Transform transform;

  transform = Transform::Translate(0.1, -0.2, 0.3) *
      Transform::Rotate(0.9866, 0.0930, -0.0969, 0.0930);

  frame.Tcw = transform;

  Projection projection;
  projection.SetFocalLength(547, 547);
  projection.SetCenterPoint(320, 240);
  frame.projection = projection;

  {
    std::shared_ptr<Image> depth_image;
    depth_image = std::make_shared<Image>(w, h);
    thrust::host_vector<float> depths(w * h);

    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        depths[y * w + x] = 1;
        depths[y * w + x] += 0.05 * cos(16 * M_PI * x / (w - 1));
        depths[y * w + x] += 0.05 * cos(16 * M_PI * y / (h - 1));
      }
    }

    thrust::device_ptr<float> ptr(depth_image->GetData());
    thrust::copy(depths.begin(), depths.end(), ptr);
    frame.depth_image = depth_image;
  }

  {
    std::shared_ptr<ColorImage> color_image;
    color_image = std::make_shared<ColorImage>(w, h);
    thrust::host_vector<Vector3f> colors(w * h);

    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        colors[y * w + x] = Vector3f(0.1, 0.2, 0.3);
      }
    }

    thrust::device_ptr<Vector3f> ptr(color_image->GetData());
    thrust::copy(colors.begin(), colors.end(), ptr);
    frame.color_image = color_image;
  }

  std::shared_ptr<ColorImage> normal_image;
  normal_image = std::make_shared<ColorImage>(w, h);
  frame.normal_image = normal_image;
  frame.ComputeNormals();
}

inline void ComputeResiduals(const Transform& Tmc, const float* keyframe_depths,
    const Vector3f* keyframe_normals, const Projection& keyframe_projection,
    int keyframe_width, int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const Projection& frame_projection,
    int frame_width, int frame_height, std::vector<float>& residuals)
{
  for (int frame_y = 0; frame_y < frame_height; ++frame_y)
  {
    for (int frame_x = 0; frame_x < frame_width; ++frame_x)
    {
      const int frame_index = frame_y * frame_width + frame_x;
      const float frame_depth = frame_depths[frame_index];

      if (frame_depth > 0)
      {
        const float frame_u = frame_x + 0.5f;
        const float frame_v = frame_y + 0.5f;
        const Vector3f Xcp = frame_projection.Unproject(frame_u, frame_v, frame_depth);
        const Vector3f Xmp = Vector3f(Tmc * Vector4f(Xcp, 1));
        const Vector2f keyframe_uv = keyframe_projection.Project(Xmp);

        if (keyframe_uv[0] >= 0 && keyframe_uv[0] < keyframe_width &&
            keyframe_uv[1] >= 0 && keyframe_uv[1] < keyframe_height)
        {
          const int keyframe_x = keyframe_uv[0];
          const int keyframe_y = keyframe_uv[1];
          const int keyframe_index = keyframe_y * keyframe_width + keyframe_x;
          const float keyframe_depth = keyframe_depths[keyframe_index];

          if (keyframe_depth > 0)
          {
            Vector3f frame_normal = frame_normals[frame_index];
            frame_normal = Vector3f(Tmc * Vector4f(frame_normal, 0));
            const Vector3f keyframe_normal = keyframe_normals[keyframe_index];

            if (keyframe_normal.SquaredNorm() > 0 &&
                frame_normal.Dot(keyframe_normal) > 0.5f)
            {
              const Vector3f Ymp = keyframe_projection.Unproject(keyframe_uv, keyframe_depth);
              const Vector3f delta = Xmp - Ymp;

              if (delta.SquaredNorm() < 0.05)
              {
                residuals[frame_index] = delta.Dot(keyframe_normal);
              }
            }
          }
        }
      }
    }
  }
}

inline void ComputeResiduals(const Frame& keyframe, const Frame& frame,
    std::vector<float>& residuals)
{
  const int frame_width = frame.depth_image->GetWidth();
  const int frame_height = frame.depth_image->GetHeight();
  const int keyframe_width = keyframe.depth_image->GetWidth();
  const int keyframe_height = keyframe.depth_image->GetHeight();
  const int frame_total = frame_width * frame_height;
  const int keyframe_total = keyframe_width * keyframe_height;
  const Projection& frame_projection = frame.projection;
  const Projection& keyframe_projection = keyframe.projection;
  const Transform Tmc = keyframe.Tcw * frame.Tcw.Inverse();

  thrust::device_ptr<const float> d_frame_depths(frame.depth_image->GetData());
  thrust::host_vector<float> h_frame_depths(d_frame_depths, d_frame_depths + frame_total);
  const float* frame_depths = h_frame_depths.data();

  thrust::device_ptr<const float> d_keyframe_depths(keyframe.depth_image->GetData());
  thrust::host_vector<float> h_keyframe_depths(d_keyframe_depths, d_keyframe_depths + keyframe_total);
  const float* keyframe_depths = h_keyframe_depths.data();

  thrust::device_ptr<const Vector3f> d_keyframe_normals(keyframe.normal_image->GetData());
  thrust::host_vector<Vector3f> h_keyframe_normals(d_keyframe_normals, d_keyframe_normals + keyframe_total);
  const Vector3f* keyframe_normals = h_keyframe_normals.data();

  thrust::device_ptr<const Vector3f> d_frame_normals(frame.normal_image->GetData());
  thrust::host_vector<Vector3f> h_frame_normals(d_frame_normals, d_frame_normals + frame_total);
  const Vector3f* frame_normals = h_frame_normals.data();

  residuals.resize(frame_total);

  ComputeResiduals(Tmc, keyframe_depths, keyframe_normals, keyframe_projection,
      keyframe_width, keyframe_height, frame_depths, frame_normals,
      frame_projection, frame_width, frame_height, residuals);
}

TEST(DepthTracker, Residuals)
{
  DepthTracker tracker;
  Buffer<float> buffer;
  std::vector<float> found;
  std::vector<float> expected;
  std::shared_ptr<Frame> keyframe;
  Frame frame;

  keyframe = std::make_shared<Frame>();
  CreateKeyframe(*keyframe);
  tracker.SetKeyframe(keyframe);

  // exact match check

  CreateKeyframe(frame);
  tracker.ComputeResiduals(frame, buffer);
  found.resize(buffer.GetSize());
  buffer.CopyToHost(found.data());

  for (size_t i = 0; i < found.size(); ++i)
  {
    ASSERT_EQ(0, found[i]);
  }

  // actual residual check

  CreateFrame(frame);
  tracker.ComputeResiduals(frame, buffer);
  found.resize(buffer.GetSize());
  buffer.CopyToHost(found.data());

  ComputeResiduals(*keyframe, frame, expected);
  ASSERT_EQ(expected.size(), found.size());

  for (size_t i = 0; i < expected.size(); ++i)
  {
    ASSERT_NEAR(expected[i], found[i], 1E-6);
  }
}

} // namespace testing

} // namespace vulcan