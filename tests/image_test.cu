#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <vulcan/image.h>

namespace vulcan
{
namespace testing
{

inline float Sample(const cv::Mat& image, float u, float v)
{
  const int x = floorf(u - 0.5f);
  const int y = floorf(v - 0.5f);

  if (x < 0 || x >= image.cols - 1 || y < 0 || y >= image.rows - 1)
  {
    return 0;
  }

  const float wu1 = u - (x + 0.5f);
  const float wv1 = v - (y + 0.5f);
  const float wu0 = 1.0f - wu1;
  const float wv0 = 1.0f - wv1;

  float sample = 0;
  sample += wv0 * wu0 * image.at<float>(y + 0, x + 0);
  sample += wv0 * wu1 * image.at<float>(y + 0, x + 1);
  sample += wv1 * wu0 * image.at<float>(y + 1, x + 0);
  sample += wv1 * wu1 * image.at<float>(y + 1, x + 1);
  return sample;
}

TEST(Image, GetGradients)
{
  int w = 640;
  int h = 480;
  Image image(w, h);
  thrust::host_vector<float> data(image.GetTotal());

  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      float value = 0;
      value += 0.25f + 0.25f * cosf(64 * M_PI * x / (w - 1));
      value += 0.25f + 0.25f * cosf(64 * M_PI * y / (h - 1));
      data[y * w + x] = value * value;
    }
  }

  Image gx, gy;
  thrust::device_ptr<float> dst(image.GetData());
  thrust::copy(data.begin(), data.end(), dst);
  image.GetGradients(gx, gy);

  cv::Mat h_image(h, w, CV_32FC1);
  cv::Mat h_gx(h, w, CV_32FC1);
  cv::Mat h_gy(h, w, CV_32FC1);

  {
    float* dst = reinterpret_cast<float*>(h_image.data);
    thrust::device_ptr<const float> src(image.GetData());
    thrust::copy(src, src + image.GetTotal(), dst);
  }

  {
    float* dst = reinterpret_cast<float*>(h_gx.data);
    thrust::device_ptr<const float> src(gx.GetData());
    thrust::copy(src, src + gx.GetTotal(), dst);
  }

  {
    float* dst = reinterpret_cast<float*>(h_gy.data);
    thrust::device_ptr<const float> src(gy.GetData());
    thrust::copy(src, src + gy.GetTotal(), dst);
  }


  cv::Mat expected_gx, expected_gy;
  cv::Sobel(h_image, expected_gx, CV_32FC1, 1, 0, 3, 1, 0);
  cv::Sobel(h_image, expected_gy, CV_32FC1, 0, 1, 3, 1, 0);
  expected_gx /= 8.0f;
  expected_gy /= 8.0f;

  for (int y = 2; y < h - 2; ++y)
  {
    for (int x = 2; x < w - 2; ++x)
    {
      ASSERT_NEAR(expected_gx.at<float>(y, x), h_gx.at<float>(y, x), 1E-6);
      ASSERT_NEAR(expected_gy.at<float>(y, x), h_gy.at<float>(y, x), 1E-6);
    }
  }

  const float d = 1E-3f;

  for (int y = 2; y < h - 2; ++y)
  {
    for (int x = 2; x < w - 2; ++x)
    {
      const float u = x + 0.5f;
      const float v = y + 0.5f;

      const float xp = Sample(h_image, u + d, v);
      const float xm = Sample(h_image, u - d, v);
      const float expected_x = (xp - xm) / (2 * d);

      const float yp = Sample(h_image, u, v + d);
      const float ym = Sample(h_image, u, v - d);
      const float expected_y = (yp - ym) / (2 * d);

      const float found_x = Sample(h_gx, u, v);
      const float found_y = Sample(h_gy, u, v);

      const float ratio_x = fabsf((expected_x - found_x) / expected_x);
      const float ratio_y = fabsf((expected_y - found_y) / expected_y);

      const float limit_x = fabsf(0.003f / expected_x);
      const float limit_y = fabsf(0.003f / expected_y);

      ASSERT_NEAR(0, ratio_x, limit_x);
      ASSERT_NEAR(0, ratio_y, limit_y);
    }
  }
}

} // namespace testing

} // namespace vulcan