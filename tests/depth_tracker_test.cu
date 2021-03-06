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

inline void ComputeResiduals(const Transform& Twm, const Transform& Twc,
    const float* keyframe_depths, const Vector3f* keyframe_normals,
    const Projection& keyframe_projection, int keyframe_width,
    int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const Projection& frame_projection,
    int frame_width, int frame_height, std::vector<double>& residuals,
    const Transform& base_Twc)
{
  for (int frame_y = 0; frame_y < frame_height; ++frame_y)
  {
    for (int frame_x = 0; frame_x < frame_width; ++frame_x)
    {
      const int frame_index = frame_y * frame_width + frame_x;
      const double frame_depth = frame_depths[frame_index];

      Matrix3d keyframe_K = Matrix3d::Identity();
      keyframe_K(0, 0) = keyframe_projection.GetFocalLength()[0];
      keyframe_K(1, 1) = keyframe_projection.GetFocalLength()[1];
      keyframe_K(0, 2) = keyframe_projection.GetCenterPoint()[0];
      keyframe_K(1, 2) = keyframe_projection.GetCenterPoint()[1];

      Matrix3d keyframe_Kinv = Matrix3d::Identity();
      keyframe_Kinv(0, 0) = 1.0 / keyframe_projection.GetFocalLength()[0];
      keyframe_Kinv(1, 1) = 1.0 / keyframe_projection.GetFocalLength()[1];
      keyframe_Kinv(0, 2) = -keyframe_projection.GetCenterPoint()[0] * keyframe_Kinv(0, 0);
      keyframe_Kinv(1, 2) = -keyframe_projection.GetCenterPoint()[1] * keyframe_Kinv(1, 1);

      Matrix3d frame_Kinv = Matrix3d::Identity();
      frame_Kinv(0, 0) = 1.0 / frame_projection.GetFocalLength()[0];
      frame_Kinv(1, 1) = 1.0 / frame_projection.GetFocalLength()[1];
      frame_Kinv(0, 2) = -frame_projection.GetCenterPoint()[0] * frame_Kinv(0, 0);
      frame_Kinv(1, 2) = -frame_projection.GetCenterPoint()[1] * frame_Kinv(1, 1);

      if (frame_depth > 0)
      {
        const double frame_u = frame_x + 0.5;
        const double frame_v = frame_y + 0.5;

        const Vector3d Xcp = frame_depth * frame_Kinv * Vector3d(frame_u, frame_v, 1);
        const Vector3d Xwp = Vector3d(Matrix4d(Twc.GetMatrix()) * Vector4d(Xcp, 1));
        const Vector3d base_Xwp = Vector3d(Matrix4d(base_Twc.GetMatrix()) * Vector4d(Xcp, 1));
        const Vector3d base_Xmp = Vector3d(Matrix4d(Twm.GetInverseMatrix()) * Vector4d(base_Xwp, 1));

        const Vector3d h_keyframe_uv = keyframe_K * base_Xmp;
        const Vector2d keyframe_uv = Vector2d(h_keyframe_uv) / h_keyframe_uv[2];

        if (keyframe_uv[0] >= 0 && keyframe_uv[0] < keyframe_width &&
            keyframe_uv[1] >= 0 && keyframe_uv[1] < keyframe_height)
        {
          const int keyframe_x = keyframe_uv[0];
          const int keyframe_y = keyframe_uv[1];
          const int keyframe_index = keyframe_y * keyframe_width + keyframe_x;
          const double keyframe_depth = keyframe_depths[keyframe_index];

          if (keyframe_depth > 0)
          {
            const Vector3d frame_normal_x = Vector3d(frame_normals[frame_index]);
            const Vector3d frame_normal = Vector3d(Matrix4d(Twc.GetMatrix()) * Vector4d(frame_normal_x, 0));

            const Vector3d keyframe_normal_x = Vector3d(keyframe_normals[keyframe_index]);
            const Vector3d keyframe_normal = Vector3d(Matrix4d(Twm.GetMatrix()) * Vector4d(keyframe_normal_x, 0));

            if (keyframe_normal.SquaredNorm() > 0 &&
                frame_normal.Dot(keyframe_normal) > 0.5)
            {
              Vector2d final_keyframe_uv;
              final_keyframe_uv[0] = keyframe_x + 0.5;
              final_keyframe_uv[1] = keyframe_y + 0.5;
              const Vector3d Ymp = keyframe_depth * keyframe_Kinv * Vector3d(final_keyframe_uv, 1);
              const Vector3d Ywp = Vector3d(Matrix4d(Twm.GetMatrix()) * Vector4d(Ymp, 1));
              const Vector3d delta = Xwp - Ywp;
              const Vector3d base_delta = base_Xwp - Ywp;

              if (base_delta.SquaredNorm() < 0.05)
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
    std::vector<double>& residuals, const Transform& base_Twc)
{
  const int frame_width = frame.depth_image->GetWidth();
  const int frame_height = frame.depth_image->GetHeight();
  const int keyframe_width = keyframe.depth_image->GetWidth();
  const int keyframe_height = keyframe.depth_image->GetHeight();
  const int frame_total = frame_width * frame_height;
  const int keyframe_total = keyframe_width * keyframe_height;
  const Projection& frame_projection = frame.projection;
  const Projection& keyframe_projection = keyframe.projection;
  const Transform& Twm = keyframe.Twc;
  const Transform& Twc = frame.Twc;

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

  ComputeResiduals(Twm, Twc, keyframe_depths, keyframe_normals,
      keyframe_projection, keyframe_width, keyframe_height, frame_depths,
      frame_normals, frame_projection, frame_width, frame_height, residuals,
      base_Twc);
}

inline Transform GetTransform(const Vector6f& update, const Transform& Twc)
{
  Matrix4f Tinc;

  Tinc(0, 0) = 1.0;
  Tinc(0, 1) = -update[2];
  Tinc(0, 2) = +update[1];
  Tinc(0, 3) = +update[3];

  Tinc(1, 0) = +update[2];
  Tinc(1, 1) = 1.0;
  Tinc(1, 2) = +update[0];
  Tinc(1, 3) = +update[4];

  Tinc(2, 0) = -update[1];
  Tinc(2, 1) = +update[0];
  Tinc(2, 2) = 1.0;
  Tinc(2, 3) = +update[5];

  Tinc(3, 0) = 0.0;
  Tinc(3, 1) = 0.0;
  Tinc(3, 2) = 0.0;
  Tinc(3, 3) = 1.0;

  const Matrix4f M = Tinc * Twc.GetMatrix();

  Vector3f x_axis(M(0, 0), M(1, 0), M(2, 0));
  Vector3f y_axis(M(0, 1), M(1, 1), M(2, 1));
  Vector3f z_axis(M(0, 2), M(1, 2), M(2, 2));

  x_axis.Normalize();
  y_axis.Normalize();
  z_axis = x_axis.Cross(y_axis);
  y_axis = z_axis.Cross(x_axis);

  Matrix3f R;

  R(0, 0) = x_axis[0];
  R(1, 0) = x_axis[1];
  R(2, 0) = x_axis[2];

  R(0, 1) = y_axis[0];
  R(1, 1) = y_axis[1];
  R(2, 1) = y_axis[2];

  R(0, 2) = z_axis[0];
  R(1, 2) = z_axis[1];
  R(2, 2) = z_axis[2];

  Vector3f t;

  t[0] = M(0, 3);
  t[1] = M(1, 3);
  t[2] = M(2, 3);

  return Transform::Translate(t) * Transform::Rotate(R);
}

inline void ComputeJacobian(const Frame& keyframe, Frame& frame,
    std::vector<Vector6f>& jacobian)
{
  std::vector<double> step_sizes(6);
  step_sizes[0] = 1E-2;
  step_sizes[1] = 1E-2;
  step_sizes[2] = 1E-2;
  step_sizes[3] = 1E-3;
  step_sizes[4] = 1E-3;
  step_sizes[5] = 1E-3;

  std::vector<double> add_residuals;
  std::vector<double> sub_residuals;
  const Transform base_Twc = frame.Twc;
  const int w = frame.depth_image->GetWidth();
  const int h = frame.depth_image->GetHeight();
  jacobian.resize(w * h);

  for (int i = 0; i < 6; ++i)
  {
    Vector6f transform = Vector6f::Zeros();

    transform[i] = +step_sizes[i];
    frame.Twc = GetTransform(transform, base_Twc);
    ComputeResiduals(keyframe, frame, add_residuals, base_Twc);

    transform[i] = -step_sizes[i];
    frame.Twc = GetTransform(transform, base_Twc);
    ComputeResiduals(keyframe, frame, sub_residuals, base_Twc);

    for (size_t j = 0; j < jacobian.size(); ++j)
    {
      const double add = add_residuals[j];
      const double sub = sub_residuals[j];
      jacobian[j][i] = (add - sub) / (2 * step_sizes[i]);
    }
  }
}

TEST(DepthTracker, Jacobian)
{
  DepthTracker tracker;
  tracker.SetTranslationEnabled(true);

  Buffer<Vector6f> buffer;
  std::vector<Vector6f> found;
  std::vector<Vector6f> expected;
  std::shared_ptr<Frame> keyframe;
  Frame frame;

  keyframe = std::make_shared<Frame>();
  CreateKeyframe(*keyframe);
  tracker.SetKeyframe(keyframe);
  CreateFrame(frame);

  tracker.ComputeJacobian(frame, buffer);
  found.resize(buffer.GetSize());
  buffer.CopyToHost(found.data());

  ComputeJacobian(*keyframe, frame, expected);
  ASSERT_EQ(expected.size(), found.size());

  const float epsilon = 7E-4;

  for (size_t i = 0; i < expected.size(); ++i)
  {
    ASSERT_NEAR(expected[i][0], found[i][0], epsilon);
    ASSERT_NEAR(expected[i][1], found[i][1], epsilon);
    ASSERT_NEAR(expected[i][2], found[i][2], epsilon);
    ASSERT_NEAR(expected[i][3], found[i][3], epsilon);
    ASSERT_NEAR(expected[i][4], found[i][4], epsilon);
    ASSERT_NEAR(expected[i][5], found[i][5], epsilon);
  }
}

TEST(DepthTracker, Residuals)
{
  DepthTracker tracker;
  Buffer<float> buffer;
  std::vector<float> found;
  std::vector<double> expected;
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

  ComputeResiduals(*keyframe, frame, expected, frame.Twc);
  ASSERT_EQ(expected.size(), found.size());

  for (size_t i = 0; i < expected.size(); ++i)
  {
    ASSERT_NEAR(expected[i], found[i], 1E-6);
  }
}

} // namespace testing

} // namespace vulcan