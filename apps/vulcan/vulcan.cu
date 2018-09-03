#include <iomanip>
#include <sstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <HAL/Camera/CameraDevice.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <vulcan/vulcan.h>

DEFINE_string(cam, "", "HAL camera uri");
DEFINE_int32(main_blocks, 65024, "main block count");
DEFINE_int32(excess_blocks, 8192, "excess block count");
DEFINE_double(voxel_length, 0.008, "voxel edge length");
DEFINE_double(trunc_length, 0.04, "volume truncation length");
DEFINE_double(min_depth, 0.1, "minimum depth value to process");
DEFINE_double(max_depth, 5.0, "maximum depth value to process");
DEFINE_int32(start_frame, 0, "frame number to begin reconstruction");
DEFINE_int32(stop_frame, 1, "frame number to end reconstruction");
DEFINE_string(output, "output.ply", "output mesh file");

#include <ctime>

using namespace vulcan;

struct Response
{
  inline float operator()(float value) const
  {
    float pow = value;
    float result = float(0);

    for (int i = 0; i < coeffs.size(); ++i)
    {
      result += pow * coeffs[i];
      pow *= value;
    }

    return result;
  }

  Eigen::VectorXf coeffs;
};

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  VULCAN_ASSERT_MSG(!FLAGS_cam.empty(), "missing camera uri");

  LOG(INFO) << "Creating camera...";

  std::shared_ptr<hal::Camera> camera;
  camera = std::make_shared<hal::Camera>(FLAGS_cam);

  Response response;
  response.coeffs = Eigen::Vector3f(0.504067,  0.0248056,  0.471128);

  cv::Mat vignetting = cv::imread("/home/mike/Code/photocalib/build/apps/vignetting/xtion_vignetting_smooth.png", CV_LOAD_IMAGE_ANYDEPTH);
  vignetting.convertTo(vignetting, CV_32FC1, 1.0 / 65535.0);

  LOG(INFO) << "Creating volume...";

  const Vector2f depth_range(FLAGS_min_depth, FLAGS_max_depth);

  std::shared_ptr<Volume> volume;
  volume = std::make_shared<Volume>(FLAGS_main_blocks, FLAGS_excess_blocks);
  volume->SetTruncationLength(FLAGS_trunc_length);
  volume->SetVoxelLength(FLAGS_voxel_length);
  volume->SetDepthRange(depth_range);

  LOG(INFO) << "Creating integrator...";

  // DepthIntegrator integrator(volume);
  // integrator.SetDepthRange(depth_range);
  // integrator.SetMaxDistanceWeight(32);

  // ColorIntegrator integrator(volume);
  // integrator.SetDepthRange(depth_range);
  // integrator.SetMaxDistanceWeight(200);
  // integrator.SetMaxColorWeight(200);

  Light light;
  // light.SetIntensity(2.0f);
  // light.SetPosition(0.1f, 0.0f, 0.0f);
  light.SetIntensity(2.0f);
  light.SetPosition(0.025f, 0.080f, 0.00f);

  LightIntegrator integrator(volume);
  integrator.SetDepthRange(depth_range);
  integrator.SetMaxDistanceWeight(100);
  integrator.SetMaxColorWeight(16);
  integrator.SetLight(light);

  LOG(INFO) << "Creating tracer...";

  Tracer tracer(volume);
  tracer.SetDepthRange(depth_range);

  LOG(INFO) << "Creating tracker...";

  // PyramidTracker<DepthTracker> tracker;
  // PyramidTracker<ColorTracker> tracker;

  std::shared_ptr<LightTracker> light_tracker;
  light_tracker = std::make_shared<LightTracker>();
  light_tracker->SetLight(light);
  // PyramidTracker<LightTracker> tracker(light_tracker);
  LightTracker& tracker = *light_tracker;
  tracker.SetMaxIterations(1);

  LOG(INFO) << "Creating tracing frame...";

  int w = 640;
  int h = 480;

  // // Vector3f tdc(0.02638751, 4.578596e-06, 0.0032462);
  // // Vector3f tdc(-0.02, 0.0, 0.0);
  // Vector3f tdc(0.0, 0.0, 0.0);

  // Matrix3f Rdc;

  // // Rdc(0, 0) =  0.999961900;
  // // Rdc(1, 0) = -0.008240786;
  // // Rdc(2, 0) = -0.002886043;

  // // Rdc(0, 1) =  0.008251156;
  // // Rdc(1, 1) =  0.999959500;
  // // Rdc(2, 1) =  0.003599982;

  // // Rdc(0, 2) =  0.002856259;
  // // Rdc(1, 2) = -0.003623658;
  // // Rdc(2, 2) =  0.999989400;

  // Rdc = Matrix3f::Identity();

  // const Transform Tdc = Transform::Translate(tdc) * Transform::Rotate(Rdc);
  // const Transform Tcd = Tdc.Inverse();

  const Transform Tcd;

  std::shared_ptr<Frame> trace_frame;
  trace_frame = std::make_shared<Frame>();

  // trace_frame->depth_projection.SetFocalLength(547, 547);
  // trace_frame->depth_projection.SetCenterPoint(320, 240);
  // trace_frame->color_projection.SetFocalLength(547, 547);
  // trace_frame->color_projection.SetCenterPoint(320, 240);

  trace_frame->depth_projection.SetFocalLength(544.1620, 544.3847);
  trace_frame->depth_projection.SetCenterPoint(311.2701, 234.7798);
  trace_frame->color_projection.SetFocalLength(544.1620, 544.3847);
  trace_frame->color_projection.SetCenterPoint(311.2701, 234.7798);

  // trace_frame->depth_projection.SetFocalLength(587.3380, 588.0155);
  // trace_frame->depth_projection.SetCenterPoint(327.7136, 238.8891);
  // trace_frame->color_projection.SetFocalLength(587.3380, 588.0155);
  // trace_frame->color_projection.SetCenterPoint(327.7136, 238.8891);

  trace_frame->depth_to_color_transform = Transform();
  trace_frame->depth_image = std::make_shared<Image>(w, h);
  trace_frame->color_image = std::make_shared<ColorImage>(w, h);
  trace_frame->normal_image = std::make_shared<ColorImage>(w, h);

  LOG(INFO) << "Integrating frames...";

  const int frame_start = FLAGS_start_frame;
  const int frame_stop  = FLAGS_stop_frame;
  const clock_t start = clock();
  bool first_frame = true;


  std::vector<cv::Mat> temp_images;

  for (int i = 0; i < frame_start; ++i)
  {
    camera->Capture(temp_images);
  }

  for (int i = frame_start; i <= frame_stop; ++i)
  {
    // if (i == 287)
    // {
    //   light_tracker->write_ = true;
    //   std::cout << "Main::Investigating..." << std::endl;;
    // }
    // else
    // {
    //   light_tracker->write_ = false;
    // }

    const int fid = i;

    std::shared_ptr<Image> depth_image;
    depth_image = std::make_shared<Image>();

    std::shared_ptr<ColorImage> color_image;
    color_image = std::make_shared<ColorImage>();

    std::vector<cv::Mat> images;
    camera->Capture(images);

    {
      cv::cvtColor(images[0], images[0], CV_BGR2RGB);
      images[0].convertTo(images[0], CV_32FC3, 1.0 / 255.0);

      for (int y = 0; y < images[0].rows; ++y)
      {
        for (int x = 0; x < images[0].cols; ++x)
        {
          const float scale = 0.75f / vignetting.at<float>(y, x);
          Eigen::Vector3f& pixel = images[0].at<Eigen::Vector3f>(y, x);
          pixel[0] = scale * response(pixel[0]);
          pixel[1] = scale * response(pixel[1]);
          pixel[2] = scale * response(pixel[2]);
        }
      }

      color_image->Load(images[0], 1.0);
      VULCAN_ASSERT(color_image->GetHeight() == h);
      VULCAN_ASSERT(color_image->GetWidth() == w);

      std::stringstream buffer;
      buffer << "input_color_";
      buffer << std::setw(4) << std::setfill('0') << fid << ".png";
      images[0].convertTo(images[0], CV_8UC3, 255.0);
      cv::imwrite(buffer.str(), images[0]);
    }

    {
      depth_image->Load(images[1], 1.0 / 10000.0);
      VULCAN_ASSERT(depth_image->GetHeight() == h);
      VULCAN_ASSERT(depth_image->GetWidth() == w);
    }

    // {
    //   std::stringstream buffer;
    //   // // // buffer << "/home/mike/Code/spelunk/build/apps/spelunk/depth_";
    //   // // // buffer << "/home/mike/Code/spelunk/build/apps/spelunk/static/depth_";
    //   // // buffer << "/home/mike/Code/spelunk/build/apps/spelunk/dynamic/left/depth_";
    //   // // // buffer << "/home/mike/Code/spelunk/build/apps/postprocess/depth_";
    //   // // buffer << std::setw(4) << std::setfill('0') << fid << "_left.png";
    //   // buffer << "/home/mike/Datasets/Work/cornell_shark/images/depth_";
    //   // buffer << std::setw(4) << std::setfill('0') << fid << ".png";
    //   // depth_image->Load(buffer.str(), 1.0 / 1000.0);
    //   buffer << "/home/mike/Code/arpg/arpg_apps/build/applications/logtool/light_car_18/channel1_";
    //   buffer << std::setw(5) << std::setfill('0') << fid << ".pgm";
    //   depth_image->Load(buffer.str(), 1.0 / 10000.0);
    //   LOG(INFO) << "Loading depth image: " << buffer.str();
    //   VULCAN_ASSERT(depth_image->GetHeight() == h);
    //   VULCAN_ASSERT(depth_image->GetWidth() == w);
    // }

    // {
    //   std::stringstream buffer;
    //   // // buffer << "/home/mike/Code/spelunk/build/apps/spelunk/color_";
    //   // // buffer << "/home/mike/Code/spelunk/build/apps/spelunk/static/color_";
    //   // buffer << "/home/mike/Code/spelunk/build/apps/spelunk/dynamic/left/color_";
    //   // // buffer << "/home/mike/Code/spelunk/build/apps/postprocess/color_";
    //   // buffer << std::setw(4) << std::setfill('0') << fid << "_left.png";
    //   // buffer << "/home/mike/Datasets/Work/cornell_shark/images/color_";
    //   // buffer << std::setw(4) << std::setfill('0') << fid << ".png";
    //   buffer << "/home/mike/Code/arpg/arpg_apps/build/applications/logtool/light_car_18/channel0_";
    //   buffer << std::setw(5) << std::setfill('0') << fid << ".pgm";
    //   LOG(INFO) << "Loading color image: " << buffer.str();
    //   color_image->Load(buffer.str(), 1.0 / 255.0);
    //   VULCAN_ASSERT(color_image->GetHeight() == h);
    //   VULCAN_ASSERT(color_image->GetWidth() == w);
    // }

    Frame frame;

    // // old IR (and unknown) calibration
    // frame.depth_projection.SetFocalLength(587.3380, 588.0155);
    // frame.depth_projection.SetCenterPoint(327.7136, 238.8891);

    // // as per online specs
    // frame.depth_projection.SetFocalLength(577.2953, 579.4113);
    // frame.depth_projection.SetCenterPoint(320.0000, 240.0000);

    // copy of color calibration
    frame.depth_projection.SetFocalLength(544.1620, 544.3847);
    frame.depth_projection.SetCenterPoint(311.2701, 234.7798);

    frame.color_projection.SetFocalLength(544.1620, 544.3847);
    frame.color_projection.SetCenterPoint(311.2701, 234.7798);

    // frame.depth_projection.SetFocalLength(547, 547);
    // frame.depth_projection.SetCenterPoint(320, 240);
    // frame.color_projection.SetFocalLength(547, 547);
    // frame.color_projection.SetCenterPoint(320, 240);
    frame.depth_to_color_transform = Tcd;
    frame.color_image = color_image;
    frame.depth_image = depth_image;
    // frame.FilterDepths();
    frame.ComputeNormals();

    if (first_frame)
    {
      first_frame = false;

      frame.normal_image->Save("given_normals.png", CV_8UC3, 127.5, 127.5);
    }
    else
    {
      LOG(INFO) << "Tracking frame " << i << "...";
      frame.depth_to_world_transform = trace_frame->depth_to_world_transform;
      tracker.SetKeyframe(trace_frame);
      tracker.Track(frame);

      LOG(INFO) << "Current pose:" << std::endl << frame.depth_to_world_transform.GetMatrix() << std::endl;
    }

    LOG(INFO) << "Viewing frame " << i << "...";
    volume->SetView(frame);
    volume->SetView(frame);
    volume->SetView(frame);

    LOG(INFO) << "Integrating frame " << i << "...";
    integrator.Integrate(frame);

    LOG(INFO) << "Tracing frame " << i << "...";
    trace_frame->depth_to_world_transform = frame.depth_to_world_transform;
    tracer.Trace(*trace_frame);

    if (i == 5)
      trace_frame->normal_image->Save("traced_normals.png", CV_8UC3, 127.5, 127.5);

    {
      LOG(INFO) << "Writing depth image...";
      std::stringstream buffer;
      buffer << "depth_" << std::setw(4) << std::setfill('0') << fid << ".png";
      std::shared_ptr<Image> dimage = trace_frame->depth_image;
      thrust::device_ptr<float> d_data(dimage->GetData());
      thrust::host_vector<float> data(d_data, d_data + dimage->GetTotal());
      cv::Mat image(h, w, CV_32FC1, data.data());
      image.convertTo(image, CV_16UC1, 1000);
      cv::imwrite(buffer.str(), image);
    }

    {
      LOG(INFO) << "Writing color image...";
      std::stringstream buffer;
      buffer << "color_" << std::setw(4) << std::setfill('0') << fid << ".png";
      std::shared_ptr<ColorImage> cimage = trace_frame->color_image;
      thrust::device_ptr<Vector3f> d_data(cimage->GetData());
      thrust::host_vector<Vector3f> data(d_data, d_data + cimage->GetTotal());
      cv::Mat image(h, w, CV_32FC3, data.data());
      image.convertTo(image, CV_8UC3, 255);
      cv::cvtColor(image, image, CV_BGR2RGB);
      cv::imwrite(buffer.str(), image);
    }

    {
      LOG(INFO) << "Writing normal image...";
      std::stringstream buffer;
      buffer << "normal_" << std::setw(4) << std::setfill('0') << fid << ".png";
      std::shared_ptr<ColorImage> nimage = trace_frame->normal_image;
      thrust::device_ptr<Vector3f> d_data(nimage->GetData());
      thrust::host_vector<Vector3f> data(d_data, d_data + nimage->GetTotal());
      cv::Mat image(h, w, CV_32FC3, data.data());
      image.convertTo(image, CV_8UC3, 127.5, 127.5);
      cv::cvtColor(image, image, CV_BGR2RGB);
      cv::imwrite(buffer.str(), image);
    }
  }

  CUDA_ASSERT(cudaDeviceSynchronize());

  const clock_t stop = clock();
  const double time = double(stop - start) / (CLOCKS_PER_SEC * frame_stop);
  std::cout << "Time: " << time << std::endl;
  std::cout << "FPS:  " << 1 / time << std::endl;

  LOG(INFO) << "Success";
  return 0;
}