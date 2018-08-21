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

  Integrator integrator(volume);

  LOG(INFO) << "Creating tracer...";

  Tracer tracer(volume);

  LOG(INFO) << "Creating tracker...";

  Tracker tracker;

  LOG(INFO) << "Reading frames...";

  const int frame_offset = 210;
  std::vector<Frame> frames(16);

  int w = 0;
  int h = 0;

  for (size_t i = 0; i < frames.size(); ++i)
  {
    const int fid = frame_offset + i;

    std::shared_ptr<Image> depth_image;
    depth_image = std::make_shared<Image>();

    std::shared_ptr<ColorImage> color_image;
    color_image = std::make_shared<ColorImage>();

    {
      std::stringstream buffer;
      buffer << "/home/mike/Code/spelunk/build/apps/spelunk/depth_";
      // buffer << "/home/mike/Code/spelunk/build/apps/postprocess/depth_";
      buffer << std::setw(4) << std::setfill('0') << fid << "_left.png";
      LOG(INFO) << "Loading depth image: " << buffer.str();
      depth_image->Load(buffer.str(), 1.0 / 1000.0);
    }

    if (i == 0)
    {
      w = depth_image->GetWidth();
      h = depth_image->GetHeight();
    }

    {
      std::stringstream buffer;
      buffer << "/home/mike/Code/spelunk/build/apps/spelunk/color_";
      // buffer << "/home/mike/Code/spelunk/build/apps/postprocess/color_";
      buffer << std::setw(4) << std::setfill('0') << fid << "_left.png";
      LOG(INFO) << "Loading color image: " << buffer.str();
      color_image->Load(buffer.str(), 1.0 / 255.0);
      VULCAN_ASSERT(color_image->GetHeight() == h);
      VULCAN_ASSERT(color_image->GetWidth() == w);
    }


    frames[i].Tcw = Transform::Translate(0, 0, 0);
    frames[i].projection.SetFocalLength(Vector2f(547, 547));
    frames[i].projection.SetCenterPoint(Vector2f(320, 240));
    frames[i].color_image = color_image;
    frames[i].depth_image = depth_image;
  }

  std::shared_ptr<Frame> trace_frame;
  trace_frame = std::make_shared<Frame>();
  trace_frame->Tcw = Transform::Translate(0, 0, 0);
  trace_frame->projection.SetFocalLength(Vector2f(547, 547));
  trace_frame->projection.SetCenterPoint(Vector2f(320, 240));
  trace_frame->depth_image = std::make_shared<Image>(w, h);
  trace_frame->color_image = std::make_shared<ColorImage>(w, h);
  trace_frame->normal_image = std::make_shared<ColorImage>(w, h);

  LOG(INFO) << "Integrating frames...";

  const clock_t start = clock();
  bool first_frame = true;

  for (Frame& frame : frames)
  {
    if (first_frame)
    {
      first_frame = false;
    }
    else
    {
      frame.Tcw = trace_frame->Tcw;
      tracker.SetKeyframe(trace_frame);
      tracker.SetTranslationEnabled(true);
      tracker.Track(frame);
    }

    volume->SetView(frame);
    integrator.Integrate(frame);
    trace_frame->Tcw = frame.Tcw;
    tracer.Trace(*trace_frame);
  }

  CUDA_ASSERT(cudaDeviceSynchronize());

  const clock_t stop = clock();
  const double time = double(stop - start) / (CLOCKS_PER_SEC * frames.size());
  LOG(INFO) << "Time: " << time << std::endl;
  LOG(INFO) << "FPS:  " << 1 / time << std::endl;

  {
    std::shared_ptr<Image> dimage = trace_frame->depth_image;
    thrust::device_ptr<float> d_data(dimage->GetData());
    thrust::host_vector<float> data(d_data, d_data + dimage->GetTotal());
    cv::Mat image(h, w, CV_32FC1, data.data());
    image.convertTo(image, CV_16UC1, 1000);
    cv::imwrite("depth.png", image);
  }

  {
    std::shared_ptr<ColorImage> cimage = trace_frame->color_image;
    thrust::device_ptr<Vector3f> d_data(cimage->GetData());
    thrust::host_vector<Vector3f> data(d_data, d_data + cimage->GetTotal());
    cv::Mat image(h, w, CV_32FC3, data.data());
    image.convertTo(image, CV_8UC3, 255);
    cv::cvtColor(image, image, CV_BGR2RGB);
    cv::imwrite("color.png", image);
  }

  {
    std::shared_ptr<ColorImage> nimage = trace_frame->normal_image;
    thrust::device_ptr<Vector3f> d_data(nimage->GetData());
    thrust::host_vector<Vector3f> data(d_data, d_data + nimage->GetTotal());
    cv::Mat image(h, w, CV_32FC3, data.data());
    image.convertTo(image, CV_8UC3, 127.5, 127.5);
    cv::cvtColor(image, image, CV_BGR2RGB);
    cv::imwrite("normal.png", image);
  }

  LOG(INFO) << "Success";
  return 0;
}