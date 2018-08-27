#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <vulcan/light_tracker.h>
#include <vulcan/frame.h>

namespace vulcan
{
namespace testing
{

inline void CreateKeyframeY(Frame& frame, const Light& light,
    bool compute_intensities = false)
{
  int w = 640;
  int h = 480;

  Transform transform;

  transform = Transform::Translate(0.0011, -0.0019, -0.5531) *
      Transform::Rotate(0.9998715, 0.0086385, -0.0103759, 0.0086385);

  frame.Twc = transform;
  const Vector3f origin = frame.Twc.GetTranslation();

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
        const float u = x + 0.5f;
        const float v = y + 0.5f;
        Vector3f Xcp = projection.Unproject(u, v);
        const Vector3f dir = Vector3f(frame.Twc * Vector4f(Xcp, 0));
        const float length = (1 - origin[2]) / dir[2];
        const Vector3f Xwp = origin + length * dir;
        Xcp = Vector3f(frame.Twc.Inverse() * Vector4f(Xwp, 1));
        depths[y * w + x] = Xcp[2];
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
        const float u = x + 0.5f;
        const float v = y + 0.5f;
        Vector3f Xcp = projection.Unproject(u, v);
        const Vector3f dir = Vector3f(frame.Twc * Vector4f(Xcp, 0));
        const float length = (1 - origin[2]) / dir[2];
        const Vector3f Xwp = origin + length * dir;
        Xcp = Vector3f(frame.Twc.Inverse() * Vector4f(Xwp, 1));
        const Vector3f normal = Vector3f(frame.Twc.Inverse() * Vector4f(0, 0, -1, 0));
        const float xx = Xwp[0];
        const float yy = Xwp[1];

        Vector3f color(0.5, 0.5, 0.5);
        color[0] += 0.245 * cosf(3.0 * M_PI * xx);
        color[1] += 0.245 * cosf(3.0 * M_PI * xx);
        color[2] += 0.245 * cosf(3.0 * M_PI * xx);
        color[0] += 0.245 * cosf(3.0 * M_PI * yy);
        color[1] += 0.245 * cosf(3.0 * M_PI * yy);
        color[2] += 0.245 * cosf(3.0 * M_PI * yy);

        if (compute_intensities)
        {
          color *= light.GetShading(Xcp, normal);
        }

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

inline void CreateFrameY(Frame& frame, const Light& light)
{
  int w = 640;
  int h = 480;

  Transform transform;

  transform = Transform::Translate(0.0010, -0.002, -0.4030) *
      Transform::Rotate(0.9998719, 0.0085884, -0.0104268, 0.0085884);

  frame.Twc = transform;
  const Vector3f origin = frame.Twc.GetTranslation();

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
        const float u = x + 0.5f;
        const float v = y + 0.5f;
        Vector3f Xcp = projection.Unproject(u, v);
        const Vector3f dir = Vector3f(frame.Twc * Vector4f(Xcp, 0));
        const float length = (1 - origin[2]) / dir[2];
        const Vector3f Xwp = origin + length * dir;
        Xcp = Vector3f(frame.Twc.Inverse() * Vector4f(Xwp, 1));
        depths[y * w + x] = Xcp[2];
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
        const float u = x + 0.5f;
        const float v = y + 0.5f;
        Vector3f Xcp = projection.Unproject(u, v);
        const Vector3f dir = Vector3f(frame.Twc * Vector4f(Xcp, 0));
        const float length = (1 - origin[2]) / dir[2];
        const Vector3f Xwp = origin + length * dir;
        Xcp = Vector3f(frame.Twc.Inverse() * Vector4f(Xwp, 1));
        const Vector3f normal = Vector3f(frame.Twc.Inverse() * Vector4f(0, 0, -1, 0));
        const float xx = Xwp[0];
        const float yy = Xwp[1];

        Vector3f color(0.5, 0.5, 0.5);
        color[0] += 0.245 * cosf(3.0 * M_PI * xx);
        color[1] += 0.245 * cosf(3.0 * M_PI * xx);
        color[2] += 0.245 * cosf(3.0 * M_PI * xx);
        color[0] += 0.245 * cosf(3.0 * M_PI * yy);
        color[1] += 0.245 * cosf(3.0 * M_PI * yy);
        color[2] += 0.245 * cosf(3.0 * M_PI * yy);

        color *= light.GetShading(Xcp, normal);
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

inline double SampleY(int w, int h, const float* values, double u, double v)
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

double ComputeResidualY(int keyframe_x, int keyframe_y, const Transform& Tcm,
    const float* keyframe_depths, const Vector3f* keyframe_normals,
    const float* keyframe_albedos, const Projection& keyframe_projection,
    int keyframe_width, int keyframe_height, const float* frame_depths,
    const Vector3f* frame_normals, const float* frame_intensities,
    const Projection& frame_projection, int frame_width, int frame_height,
    const Light& light, bool* visible = nullptr)
{
  VULCAN_DEBUG(keyframe_x >= 0 && keyframe_x < keyframe_width);
  VULCAN_DEBUG(keyframe_y >= 0 && keyframe_y < keyframe_height);

  if (visible) *visible = false;

  const int keyframe_index = keyframe_y * keyframe_width + keyframe_x;
  const float keyframe_depth = keyframe_depths[keyframe_index];

  if (keyframe_depth > 0.001)
  {
    const float keyframe_u = keyframe_x + 0.5;
    const float keyframe_v = keyframe_y + 0.5;
    const Vector3f Xmp = keyframe_projection.Unproject(keyframe_u, keyframe_v, keyframe_depth);
    const Vector3f Xcp = Vector3f(Tcm * Vector4f(Xmp, 1));
    const Vector2f frame_uv = frame_projection.Project(Xcp);

    if (frame_uv[0] >= 1.0 && frame_uv[0] < frame_width  - 1.0 &&
        frame_uv[1] >= 1.0 && frame_uv[1] < frame_height - 1.0)
    {
      const int frame_x = frame_uv[0];
      const int frame_y = frame_uv[1];
      const int frame_index = frame_y * frame_width + frame_x;
      const double frame_depth = frame_depths[frame_index];

      if (fabs(frame_depth - Xcp[2]) < 0.099)
      {
        const Vector3f frame_normal = frame_normals[frame_index];
        Vector3f keyframe_normal = keyframe_normals[keyframe_index];
        keyframe_normal = Vector3f(Tcm * Vector4f(keyframe_normal, 0));

        if (keyframe_normal.SquaredNorm() > 0.501 &&
            frame_normal.Dot(keyframe_normal) > 0.501)
        {
          const float aa = keyframe_albedos[keyframe_index];

          if (aa > 0)
          {
            const double shading = light.GetShading(Xcp, keyframe_normal);
            const double Im = shading * aa;

            const double Ic = SampleY(frame_width, frame_height,
                frame_intensities, frame_uv[0], frame_uv[1]);

            if (visible) *visible = true;
            return Ic - Im;
          }
        }
      }
    }
  }

  return 0;
}

void ComputeResidualsY(const Frame& keyframe, const Image& keyframe_intensities_,
    const Frame& frame, const Image& frame_intensities_, const Light& light,
    std::vector<double>& residuals, std::vector<bool>* visibles = nullptr)
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

  if (visibles) (*visibles).resize(keyframe_total);

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
      bool visible = false;
      const int i = y * keyframe_width + x;

      const double residual = ComputeResidualY(x, y, Tcm, keyframe_depths,
          keyframe_normals, keyframe_intensities, keyframe_projection,
          keyframe_width, keyframe_height, frame_depths, frame_normals,
          frame_intensities,  frame_projection, frame_width, frame_height,
          light, &visible);

      if (visibles) (*visibles)[i] = visible;
      residuals[i] = residual;
    }
  }
}

inline Transform GetTransformX(const Vector6f& update, const Transform& Twc)
{
  Matrix4f Tinc;

  Tinc(0, 0) = 1.0;
  Tinc(0, 1) = -update[2];
  Tinc(0, 2) = +update[1];
  Tinc(0, 3) = +update[3];

  Tinc(1, 0) = +update[2];
  Tinc(1, 1) = 1.0;
  Tinc(1, 2) = -update[0];
  Tinc(1, 3) = +update[4];

  Tinc(2, 0) = -update[1];
  Tinc(2, 1) = +update[0];
  Tinc(2, 2) = 1.0;
  Tinc(2, 3) = +update[5];

  Tinc(3, 0) = 0.0;
  Tinc(3, 1) = 0.0;
  Tinc(3, 2) = 0.0;
  Tinc(3, 3) = 1.0;

  const Matrix4f M = Tinc * Twc.GetInverseMatrix();

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

  return (Transform::Translate(t) * Transform::Rotate(R)).Inverse();
}

inline void ComputeJacobianY(const Frame& keyframe, Frame& frame,
    const Light& light, std::vector<Vector6f>& jacobian,
    std::vector<Vector6f>& visibles)
{
  std::vector<double> step_sizes(6);
  step_sizes[0] = 1E-2;
  step_sizes[1] = 1E-2;
  step_sizes[2] = 1E-2;
  step_sizes[3] = 1E-2;
  step_sizes[4] = 1E-2;
  step_sizes[5] = 1E-2;

  std::vector<double> add_residuals;
  std::vector<double> sub_residuals;
  const Transform base_Twc = frame.Twc;
  const int w = frame.depth_image->GetWidth();
  const int h = frame.depth_image->GetHeight();
  jacobian.resize(w * h);
  visibles.resize(w * h);

  Image keyframe_intensities;
  keyframe.color_image->ConvertTo(keyframe_intensities);

  Image frame_intensities;
  frame.color_image->ConvertTo(frame_intensities);

  std::vector<bool> add_visibles;
  std::vector<bool> sub_visibles;

  for (int i = 0; i < 6; ++i)
  {
    Vector6f transform = Vector6f::Zeros();

    transform[i] = +step_sizes[i];
    frame.Twc = GetTransformX(transform, base_Twc);

    ComputeResidualsY(keyframe, keyframe_intensities, frame, frame_intensities,
        light, add_residuals, &add_visibles);

    transform[i] = -step_sizes[i];
    frame.Twc = GetTransformX(transform, base_Twc);

    ComputeResidualsY(keyframe, keyframe_intensities, frame, frame_intensities,
        light, sub_residuals, &sub_visibles);

    for (size_t j = 0; j < jacobian.size(); ++j)
    {
      const double add = add_residuals[j];
      const double sub = sub_residuals[j];

      jacobian[j][i] = (add - sub) / (2 * step_sizes[i]);
      visibles[j][i] = (add_visibles[j] && sub_visibles[j]) ? 1 : 0;
    }
  }
}

TEST(LightTracker, Jacobian)
{
  LightTracker tracker;
  tracker.SetTranslationEnabled(true);

  Buffer<Vector6f> buffer;
  std::vector<Vector6f> found;
  std::vector<Vector6f> expected;
  std::shared_ptr<Frame> keyframe;
  Frame frame;

  Light light;
  light.SetIntensity(2.0f);
  light.SetPosition(0.1f, 0.0f, 0.0f);

  keyframe = std::make_shared<Frame>();
  CreateKeyframeY(*keyframe, light);
  tracker.SetKeyframe(keyframe);
  tracker.SetLight(light);
  CreateFrameY(frame, light);


  tracker.ComputeJacobian(frame, buffer);
  found.resize(buffer.GetSize());
  buffer.CopyToHost(found.data());

  std::vector<Vector6f> visibles;
  ComputeJacobianY(*keyframe, frame, light, expected, visibles);
  ASSERT_EQ(expected.size(), found.size());

  const int w = keyframe->depth_image->GetWidth();
  const int h = keyframe->depth_image->GetHeight();

  for (int y = 5; y < h - 5; ++y)
  {
    for (int x = 5; x < w - 5; ++x)
    {
      const int i = y * w + x;

      const float u = x + 0.5f;
      const float v = y + 0.5f;
      const float d = 1; // TODO: lookup from depth map
      const Vector3f Xmp = keyframe->projection.Unproject(u, v, d);
      const Vector3f Xwp = Vector3f(keyframe->Twc * Vector4f(Xmp, 1));
      const Vector3f Xcp = Vector3f(frame.Twc.Inverse() * Vector4f(Xwp, 1));
      const Vector2f uv = frame.projection.Project(Xcp);

      if (uv[0] < 5 || uv[0] > w - 5 || uv[1] < 5 || uv[1] > h - 5)
      {
        continue;
      }

      for (int j = 0; j < 6; ++j)
      {
        if (visibles[i][j] < 0.5) continue;

        const double f = found[i][j];
        const double e = expected[i][j];
        const double d = fabs(f - e);
        const double n = fmin(fabs(f), fabs(e));
        const double p = 0.5 * (fabs(f) + fabs(e));
        const double r = fabs(d / n);
        const double l = fabs(0.065 / p);

        // if (d > 0.05)
        // {
        //   printf("%d %d -> %f %f [%d]) e: %f, f: %f, d: %f, r: %f, l: %f\n",
        //       x, y, uv[0], uv[1], j, e, f, d, r, l);
        // }

        ASSERT_NEAR(f, e, 0.05);
      }
    }
  }
}

TEST(LightTracker, Residuals)
{
  Frame frame;
  Buffer<float> buffer;
  LightTracker tracker;
  std::shared_ptr<Frame> keyframe;
  std::vector<double> expected;
  std::vector<float> found;
  Image keyframe_intensities;
  Image frame_intensities;

  Light light;
  light.SetIntensity(2.0f);
  light.SetPosition(0.1f, 0.0f, 0.0f);

  keyframe = std::make_shared<Frame>();
  CreateKeyframeY(*keyframe, light);
  tracker.SetKeyframe(keyframe);
  tracker.SetTranslationEnabled(true);
  tracker.SetLight(light);

  // exact match check

  CreateKeyframeY(frame, light, true);
  tracker.ComputeResiduals(frame, buffer);
  found.resize(buffer.GetSize());
  buffer.CopyToHost(found.data());

  for (size_t i = 0; i < found.size(); ++i)
  {
    ASSERT_NEAR(0, found[i], 1E-4);
  }

  // actual residual check

  CreateFrameY(frame, light);
  tracker.ComputeResiduals(frame, buffer);
  found.resize(buffer.GetSize());
  buffer.CopyToHost(found.data());

  keyframe->color_image->ConvertTo(keyframe_intensities);
  frame.color_image->ConvertTo(frame_intensities);

  std::vector<bool> visibles;

  ComputeResidualsY(*keyframe, keyframe_intensities, frame, frame_intensities,
      light, expected, &visibles);

  ASSERT_EQ(expected.size(), found.size());

  for (size_t i = 0; i < expected.size(); ++i)
  {
    if (visibles[i]) ASSERT_NEAR(expected[i], found[i], 1E-4);
  }
}

TEST(LightTracker, Track)
{
  Frame frame;
  Buffer<float> buffer;
  LightTracker tracker;
  std::shared_ptr<Frame> keyframe;
  std::vector<double> expected;
  std::vector<float> found;
  Image keyframe_intensities;
  Image frame_intensities;

  Light light;
  light.SetIntensity(2.0f);
  light.SetPosition(0.1f, 0.0f, 0.0f);

  keyframe = std::make_shared<Frame>();
  CreateKeyframeY(*keyframe, light);
  tracker.SetKeyframe(keyframe);
  tracker.SetTranslationEnabled(true);
  tracker.SetLight(light);

  CreateFrameY(frame, light);

  {
    const Transform old_Twc = frame.Twc;

    for (int i = 0; i < 20; ++i)
    {
      tracker.Track(frame);
    }

    const Transform new_Twc = frame.Twc;
    const Matrix4f diff = old_Twc.GetInverseMatrix() * new_Twc.GetMatrix();

    for (int i = 0; i < diff.GetRows(); ++i)
    {
      for (int j = 0; j < diff.GetColumns(); ++j)
      {
        const float expected = (i == j) ? 1 : 0;
        ASSERT_NEAR(expected, diff(j, i), 1E-5);
      }
    }

    // std::cout << "old_Twc:" << std::endl << old_Twc.GetMatrix() << std::endl;
    // std::cout << "new_Twc:" << std::endl << new_Twc.GetMatrix() << std::endl;
    // std::cout << "dif_Twc:" << std::endl << old_Twc.GetInverseMatrix() * new_Twc.GetMatrix() << std::endl;
  }

  {
    const Transform old_Twc = frame.Twc;

    Transform prb_Twc = frame.Twc;

    prb_Twc = Transform::Translate(0.1, 0.1, 0.1) *
        Transform::Rotate(0.999871, 0.008638, -0.010375, 0.008638) * prb_Twc;

    frame.Twc = prb_Twc;

    for (int i = 0; i < 20; ++i)
    {
      tracker.Track(frame);
    }

    const Transform new_Twc = frame.Twc;
    const Matrix4f diff = old_Twc.GetInverseMatrix() * new_Twc.GetMatrix();

    for (int i = 0; i < diff.GetRows(); ++i)
    {
      for (int j = 0; j < diff.GetColumns(); ++j)
      {
        const float expected = (i == j) ? 1 : 0;
        ASSERT_NEAR(expected, diff(j, i), 1E-5);
      }
    }

    // std::cout << "old_Twc:" << std::endl << old_Twc.GetMatrix() << std::endl;
    // std::cout << "prb_Twc:" << std::endl << prb_Twc.GetMatrix() << std::endl;
    // std::cout << "new_Twc:" << std::endl << new_Twc.GetMatrix() << std::endl;
    // std::cout << "dif_Twc:" << std::endl << old_Twc.GetInverseMatrix() * new_Twc.GetMatrix() << std::endl;
  }
}

} // namespace testing

} // namespace vulcan