#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <vulcan/image.h>
#include <vulcan/sampler.h>

namespace vulcan
{
namespace testing
{

TEST(Sampler, GetSubimage)
{
  Image image, expected, found;

  image.Load("/home/mike/Code/tracking/orb_slam/Datasets/spelunk/cave_05/image_0/0000.png", 1.0 / 255.0);

  thrust::device_ptr<float> d_ptr(image.GetData());
  thrust::host_vector<float> ptr(d_ptr, d_ptr + image.GetTotal());

  Sampler sampler;
  sampler.GetSubimage(image, found, Sampler::FILTER_LINEAR);
  sampler.GetSubimage(found, expected, Sampler::FILTER_LINEAR);

  cv::Mat mat(image.GetHeight(), image.GetWidth(), CV_32FC1, ptr.data());
  cv::imshow("Image", mat);

  {
    thrust::device_ptr<float> d_ptr(found.GetData());
    thrust::host_vector<float> ptr(d_ptr, d_ptr + found.GetTotal());
    cv::Mat mat(found.GetHeight(), found.GetWidth(), CV_32FC1, ptr.data());
    cv::imshow("Found", mat);
  }

  {
    thrust::device_ptr<float> d_ptr(expected.GetData());
    thrust::host_vector<float> ptr(d_ptr, d_ptr + expected.GetTotal());
    cv::Mat mat(expected.GetHeight(), expected.GetWidth(), CV_32FC1, ptr.data());
    cv::imshow("Expected", mat);
    cv::waitKey(0);
  }
}

TEST(Sampler, GetGradient)
{
  Image image, expected, found;

  image.Load("/home/mike/Code/tracking/orb_slam/Datasets/spelunk/cave_05/image_0/0000.png", 1.0 / 255.0);

  Sampler sampler;
  // sampler.GetGradient(image, expected, found);
  sampler.GetSubimage(image, expected, Sampler::FILTER_NEAREST);
  sampler.GetSubimage(expected, image, Sampler::FILTER_NEAREST);
  sampler.GetGradient(image, expected, found);

  {
    thrust::device_ptr<float> d_ptr(image.GetData());
    thrust::host_vector<float> ptr(d_ptr, d_ptr + image.GetTotal());
    cv::Mat mat(image.GetHeight(), image.GetWidth(), CV_32FC1, ptr.data());
    cv::imshow("Image", mat);
  }

  {
    thrust::device_ptr<float> d_ptr(found.GetData());
    thrust::host_vector<float> ptr(d_ptr, d_ptr + found.GetTotal());
    cv::Mat mat(found.GetHeight(), found.GetWidth(), CV_32FC1, ptr.data());
    mat = (mat + 0.5);
    cv::imshow("Found", mat);
  }

  {
    thrust::device_ptr<float> d_ptr(expected.GetData());
    thrust::host_vector<float> ptr(d_ptr, d_ptr + expected.GetTotal());
    cv::Mat mat(expected.GetHeight(), expected.GetWidth(), CV_32FC1, ptr.data());
    mat = (mat + 0.5);
    cv::imshow("Expected", mat);
    cv::waitKey(0);
  }
}

TEST(Sampler, Profile)
{
  const int levels = 4;
  Image intensities[levels];
  Image gradients[levels][2];
  Sampler sampler;

  intensities[0].Load("/home/mike/Code/tracking/orb_slam/Datasets/spelunk/cave_05/image_0/0000.png", 1.0 / 255.0);

  const int max_iters = 1000;
  const clock_t start = clock();

  for (int i = 0; i < max_iters; ++i)
  {
    sampler.GetGradient(intensities[0], gradients[0][0], gradients[0][1]);

    for (int j = 1; j < levels; ++j)
    {
      sampler.GetSubimage(intensities[j - 1], intensities[j]);
      sampler.GetGradient(intensities[j - 1], gradients[j][0], gradients[j][1]);
    }
  }

  const clock_t stop = clock();
  const double time = double(stop - start) / (CLOCKS_PER_SEC * max_iters);
  std::cout << "Time: " << time << std::endl;
  std::cout << "FPS:  " << 1 / time << std::endl;
}

} // namespace testing

} // namespace vulcan