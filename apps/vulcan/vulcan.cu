#include <iomanip>
#include <sstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <vulcan/vulcan.h>

DEFINE_int32(main_blocks, 65024, "main block count");
DEFINE_int32(excess_blocks, 8192, "excess block count");
DEFINE_double(voxel_length, 0.008, "voxel edge length");
DEFINE_double(truncation_length, 0.04, "volume truncation length");
DEFINE_string(output, "output.ply", "output mesh file");

#include <ctime>

using namespace vulcan;

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "Creating volume...";

  std::shared_ptr<Volume> volume;
  volume = std::make_shared<Volume>(FLAGS_main_blocks, FLAGS_excess_blocks);
  volume->SetTruncationLength(FLAGS_truncation_length);
  volume->SetVoxelLength(FLAGS_voxel_length);

  LOG(INFO) << "Creating integrator...";

  Light light;
  light.SetIntensity(2.0f);
  light.SetPosition(0.1f, 0.0f, 0.0f);

  LightIntegrator integrator(volume);
  integrator.SetMaxDistanceWeight(200);
  integrator.SetMaxColorWeight(16);
  integrator.SetLight(light);

  // ColorIntegrator integrator(volume);
  // integrator.SetMaxDistanceWeight(200);
  // integrator.SetMaxColorWeight(64);

  LOG(INFO) << "Creating tracer...";

  Tracer tracer(volume);

  LOG(INFO) << "Creating tracker...";

  // PyramidTracker<DepthTracker> tracker;

  LightTracker tracker;
  tracker.SetMaxIterations(10);
  tracker.SetLight(light);

  LOG(INFO) << "Creating tracing frame...";

  int w = 640;
  int h = 480;

  std::shared_ptr<Frame> trace_frame;
  trace_frame = std::make_shared<Frame>();
  trace_frame->Tcw = Transform::Translate(0, 0, 0);
  trace_frame->projection.SetFocalLength(Vector2f(547, 547));
  trace_frame->projection.SetCenterPoint(Vector2f(320, 240));
  trace_frame->depth_image = std::make_shared<Image>(w, h);
  trace_frame->color_image = std::make_shared<ColorImage>(w, h);
  trace_frame->normal_image = std::make_shared<ColorImage>(w, h);

  LOG(INFO) << "Integrating frames...";

  const int frame_start = 10;
  const int frame_stop  = 13;
  const clock_t start = clock();
  bool first_frame = true;

  for (int i = frame_start; i <= frame_stop; ++i)
  {
    const int fid = i;

    std::shared_ptr<Image> depth_image;
    depth_image = std::make_shared<Image>();

    std::shared_ptr<ColorImage> color_image;
    color_image = std::make_shared<ColorImage>();

    {
      std::stringstream buffer;
      buffer << "/home/mike/Code/spelunk/build/apps/spelunk/depth_";
      // buffer << "/home/mike/Code/spelunk/build/apps/postprocess/depth_";
      buffer << std::setw(4) << std::setfill('0') << fid << "_left.png";
      // buffer << "/home/mike/Datasets/Work/cornell_shark/images/depth_";
      // buffer << std::setw(4) << std::setfill('0') << fid << ".png";
      LOG(INFO) << "Loading depth image: " << buffer.str();
      depth_image->Load(buffer.str(), 1.0 / 1000.0);
      VULCAN_ASSERT(depth_image->GetHeight() == h);
      VULCAN_ASSERT(depth_image->GetWidth() == w);
    }

    {
      std::stringstream buffer;
      buffer << "/home/mike/Code/spelunk/build/apps/spelunk/color_";
      // buffer << "/home/mike/Code/spelunk/build/apps/postprocess/color_";
      buffer << std::setw(4) << std::setfill('0') << fid << "_left.png";
      // buffer << "/home/mike/Datasets/Work/cornell_shark/images/color_";
      // buffer << std::setw(4) << std::setfill('0') << fid << ".png";
      LOG(INFO) << "Loading color image: " << buffer.str();
      color_image->Load(buffer.str(), 1.0 / 255.0);
      VULCAN_ASSERT(color_image->GetHeight() == h);
      VULCAN_ASSERT(color_image->GetWidth() == w);
    }

    Frame frame;
    frame.Tcw = Transform::Translate(0, 0, 0);
    frame.projection.SetFocalLength(Vector2f(547, 547));
    frame.projection.SetCenterPoint(Vector2f(320, 240));
    frame.color_image = color_image;
    frame.depth_image = depth_image;
    frame.ComputeNormals();

    if (first_frame)
    {
      first_frame = false;
    }
    else
    {
      LOG(INFO) << "Tracking frame " << i << "...";
      frame.Tcw = trace_frame->Tcw;
      tracker.SetKeyframe(trace_frame);
      tracker.Track(frame);

      LOG(INFO) << "Current pose:" << std::endl << frame.Tcw.GetMatrix() << std::endl;
    }

    LOG(INFO) << "Viewing frame " << i << "...";
    volume->SetView(frame);

    LOG(INFO) << "Integrating frame " << i << "...";
    integrator.Integrate(frame);

    LOG(INFO) << "Tracing frame " << i << "...";
    trace_frame->Tcw = frame.Tcw;
    tracer.Trace(*trace_frame);

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