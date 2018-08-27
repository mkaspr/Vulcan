#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <vulcan/color_tracker.h>
#include <vulcan/frame.h>

namespace vulcan
{
namespace testing
{

inline void CreateKeyframeX(Frame& frame)
{
  int w = 640;
  int h = 480;

  Transform transform;
  frame.Twc = transform;

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
        Vector3f color(0.1, 0.2, 0.3);
        const float xratio = float(x) / (w - 1);
        const float yratio = float(y) / (h - 1);
        color[0] += 0.3 * cosf(2.0 * M_PI * xratio);
        color[1] += 0.2 * sinf(4.0 * M_PI * xratio);
        color[2] += 0.1 * cosf(8.0 * M_PI * xratio);
        color[0] += 0.3 * cosf(2.0 * M_PI * yratio);
        color[1] += 0.2 * sinf(4.0 * M_PI * yratio);
        color[2] += 0.1 * cosf(8.0 * M_PI * yratio);
        colors[y * w + x] = color;
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

inline void CreateFrameX(Frame& frame)
{
  int w = 640;
  int h = 480;

  Transform transform;

  transform = Transform::Translate(0.001, -0.002, 0.003) *
      Transform::Rotate(0.9998719, 0.0085884, -0.0104268, 0.0085884);

  frame.Twc = transform;

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
        depths[y * w + x] += 0.01 * cos(16 * M_PI * x / (w - 1));
        depths[y * w + x] += 0.01 * cos(16 * M_PI * y / (h - 1));
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
        Vector3f color(0.1, 0.2, 0.3);
        const float xratio = float(x) / (w - 1);
        const float yratio = float(y) / (h - 1);
        color[0] += 0.3 * cosf(2.0 * M_PI * xratio);
        color[1] += 0.2 * sinf(4.0 * M_PI * xratio);
        color[2] += 0.1 * cosf(8.0 * M_PI * xratio);
        color[0] += 0.3 * cosf(2.0 * M_PI * yratio);
        color[1] += 0.2 * sinf(4.0 * M_PI * yratio);
        color[2] += 0.1 * cosf(8.0 * M_PI * yratio);
        colors[y * w + x] = color;
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

inline double Sample(int w, int h, const float* values, double u, double v)
{
  const int x = floor(u - 0.5);
  const int y = floor(v - 0.5);

  VULCAN_DEBUG(x >= 0 && x < w - 1 && y >= 0 && y < h - 1);

  const double v00 = values[(y + 0) * w + (x + 0)];
  const double v01 = values[(y + 0) * w + (x + 1)];
  const double v10 = values[(y + 1) * w + (x + 0)];
  const double v11 = values[(y + 1) * w + (x + 1)];

  const double u1 = u - (x + 0.5);
  const double v1 = v - (y + 0.5);
  const double u0 = 1.0 - u1;
  const double v0 = 1.0 - v1;

  const double w00 = v0 * u0;
  const double w01 = v0 * u1;
  const double w10 = v1 * u0;
  const double w11 = v1 * u1;

  return (w00 * v00) + (w01 * v01) + (w10 * v10) + (w11 * v11);
}

double ComputeResidual(int keyframe_x, int keyframe_y, const Transform& Tcm,
    const float* keyframe_depths, const Vector3f* keyframe_normals,
    const float* keyframe_intensities, const Projection& keyframe_projection,
    int keyframe_width, int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const float* frame_intensities,
    const Projection& frame_projection, int frame_width, int frame_height)
{
  VULCAN_DEBUG(keyframe_x >= 0 && keyframe_x < keyframe_width);
  VULCAN_DEBUG(keyframe_y >= 0 && keyframe_y < keyframe_height);

  const int keyframe_index = keyframe_y * keyframe_width + keyframe_x;
  const double keyframe_depth = keyframe_depths[keyframe_index];

  Matrix3d frame_K = Matrix3d::Identity();
  frame_K(0, 0) = frame_projection.GetFocalLength()[0];
  frame_K(1, 1) = frame_projection.GetFocalLength()[1];
  frame_K(0, 2) = frame_projection.GetCenterPoint()[0];
  frame_K(1, 2) = frame_projection.GetCenterPoint()[1];

  Matrix3d keyframe_Kinv = Matrix3d::Identity();
  keyframe_Kinv(0, 0) = 1.0 / keyframe_projection.GetFocalLength()[0];
  keyframe_Kinv(1, 1) = 1.0 / keyframe_projection.GetFocalLength()[1];
  keyframe_Kinv(0, 2) = -keyframe_projection.GetCenterPoint()[0] * keyframe_Kinv(0, 0);
  keyframe_Kinv(1, 2) = -keyframe_projection.GetCenterPoint()[1] * keyframe_Kinv(1, 1);

  if (keyframe_depth > 0)
  {
    const double keyframe_u = keyframe_x + 0.5;
    const double keyframe_v = keyframe_y + 0.5;
    const Vector3d Xmp = keyframe_depth * keyframe_Kinv * Vector3d(keyframe_u, keyframe_v, 1);
    const Vector3d Xcp = Vector3d(Matrix4d(Tcm.GetMatrix()) * Vector4d(Xmp, 1));
    const Vector3d h_frame_uv = frame_K * Xcp;
    const Vector2d frame_uv = Vector2d(h_frame_uv) / h_frame_uv[2];

    if (frame_uv[0] >= 0.5 && frame_uv[0] < frame_width  - 0.5 &&
        frame_uv[1] >= 0.5 && frame_uv[1] < frame_height - 0.5)
    {
      const int frame_x = frame_uv[0];
      const int frame_y = frame_uv[1];
      const int frame_index = frame_y * frame_width + frame_x;
      const double frame_depth = frame_depths[frame_index];

      if (fabs(frame_depth - Xcp[2]) < 0.1)
      {
        const Vector3d frame_normal = Vector3d(frame_normals[frame_index]);
        Vector3d keyframe_normal = Vector3d(keyframe_normals[keyframe_index]);
        keyframe_normal = Vector3d(Matrix4d(Tcm.GetMatrix()) * Vector4d(keyframe_normal, 0));

        if (keyframe_normal.SquaredNorm() > 0.0 &&
            frame_normal.Dot(keyframe_normal) > 0.5)
        {
          const double Im = keyframe_intensities[keyframe_index];

          const double Ic = Sample(frame_width, frame_height,
              frame_intensities, frame_uv[0], frame_uv[1]);

          return Ic - Im;
        }
      }
    }
  }

  return 0;
}

void ComputeResiduals(const Frame& keyframe, const Image& keyframe_intensities_,
    const Frame& frame, const Image& frame_intensities_,
    std::vector<double>& residuals)
{
  const int frame_width = frame.depth_image->GetWidth();
  const int frame_height = frame.depth_image->GetHeight();
  const int keyframe_width = keyframe.depth_image->GetWidth();
  const int keyframe_height = keyframe.depth_image->GetHeight();
  const Projection& frame_projection = frame.projection;
  const Projection& keyframe_projection = keyframe.projection;
  const Transform Tcm = frame.Twc.Inverse() * keyframe.Twc;

  const int frame_total = frame_width * frame_height;
  const int keyframe_total = keyframe_width * keyframe_height;

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

  thrust::device_ptr<const float> d_frame_intensities(frame_intensities_.GetData());
  thrust::host_vector<float> h_frame_intensities(d_frame_intensities, d_frame_intensities + frame_total);
  const float* frame_intensities = h_frame_intensities.data();

  thrust::device_ptr<const float> d_keyframe_intensities(keyframe_intensities_.GetData());
  thrust::host_vector<float> h_keyframe_intensities(d_keyframe_intensities, d_keyframe_intensities + keyframe_total);
  const float* keyframe_intensities = h_keyframe_intensities.data();

  residuals.resize(keyframe_total);

  for (int y = 0; y < keyframe_height; ++y)
  {
    for (int x = 0; x < keyframe_width; ++x)
    {
      const double residual = ComputeResidual(x, y, Tcm, keyframe_depths,
          keyframe_normals, keyframe_intensities, keyframe_projection,
          keyframe_width, keyframe_height, frame_depths, frame_normals,
          frame_intensities,  frame_projection, frame_width, frame_height);

      residuals[y * keyframe_width + x] = residual;
    }
  }
}

TEST(ColorTracker, Residuals)
{
  Frame frame;
  Buffer<float> buffer;
  ColorTracker tracker;
  std::shared_ptr<Frame> keyframe;
  std::vector<double> expected;
  std::vector<float> found;
  Image keyframe_intensities;
  Image frame_intensities;

  keyframe = std::make_shared<Frame>();
  CreateKeyframeX(*keyframe);
  tracker.SetKeyframe(keyframe);
  tracker.SetTranslationEnabled(true);

  // exact match check

  CreateKeyframeX(frame);
  tracker.ComputeResiduals(frame, buffer);
  found.resize(buffer.GetSize());
  buffer.CopyToHost(found.data());

  for (size_t i = 0; i < found.size(); ++i)
  {
    ASSERT_NEAR(0, found[i], 1E-6);
  }

  // actual residual check

  CreateFrameX(frame);
  tracker.ComputeResiduals(frame, buffer);
  found.resize(buffer.GetSize());
  buffer.CopyToHost(found.data());

  keyframe->color_image->ConvertTo(keyframe_intensities);
  frame.color_image->ConvertTo(frame_intensities);

  ComputeResiduals(*keyframe, keyframe_intensities, frame, frame_intensities,
      expected);

  ASSERT_EQ(expected.size(), found.size());

  for (size_t i = 0; i < expected.size(); ++i)
  {
    ASSERT_NEAR(expected[i], found[i], 1E-6);
  }
}

} // namespace testing

} // namespace vulcan