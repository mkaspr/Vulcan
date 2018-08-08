#include <iomanip>
#include <sstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <vulcan/vulcan.h>

DEFINE_int32(main_blocks, 130048, "main block count");
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

  LOG(INFO) << "Reading frames...";

  std::vector<Frame> frames(1);

  std::shared_ptr<Image> image;
  image = std::make_shared<Image>();

  std::shared_ptr<ColorImage> color_image;
  color_image = std::make_shared<ColorImage>();

  for (size_t i = 0; i < frames.size(); ++i)
  {
    {
      std::stringstream buffer;
      buffer << "/home/mike/Code/matchbox/build/apps/matchbox_live/depth_";
      buffer << std::setw(4) << std::setfill('0') << i << ".png";
      LOG(INFO) << "Loading depth image: " << buffer.str();
      image->Load(buffer.str(), 1.0 / 1000.0);
    }

    {
      std::stringstream buffer;
      buffer << "/home/mike/Code/spelunk/build/apps/spelunk/frame_";
      buffer << std::setw(4) << std::setfill('0') << i << "_left.png";
      LOG(INFO) << "Loading color image: " << buffer.str();
      color_image->Load(buffer.str(), 1.0 / 255.0);
    }

    frames[i].Tcw = Transform::Translate(0, 0, 0);
    frames[i].projection.SetFocalLength(Vector2f(547, 547));
    frames[i].projection.SetCenterPoint(Vector2f(320, 240));
    frames[i].color_image = color_image;
    frames[i].depth_image = image;
  }

  LOG(INFO) << "Creating volume...";

  std::shared_ptr<Volume> volume;
  volume = std::make_shared<Volume>(FLAGS_main_blocks, FLAGS_excess_blocks);
  volume->SetTruncationLength(FLAGS_truncation_length);
  volume->SetVoxelLength(FLAGS_voxel_length);

  LOG(INFO) << "Creating integrator...";

  Integrator integrator(volume);

  LOG(INFO) << "Integrating frames...";

  for (const Frame& frame : frames)
  {
    // REMOVE
    volume->SetView(frame);
    volume->SetView(frame);
    CUDA_ASSERT(cudaDeviceSynchronize());
    // // REMOVE

    volume->SetView(frame);
    integrator.Integrate(frame);
    CUDA_ASSERT(cudaDeviceSynchronize());
  }

  LOG(INFO) << "Tracing volume...";

  const int w = frames[0].color_image->GetWidth();
  const int h = frames[0].color_image->GetHeight();
  frames[0].normal_image = std::make_shared<ColorImage>();
  frames[0].normal_image->Resize(w, h);

  Tracer tracer(volume);
  tracer.Trace(frames[0]);

  {
    std::shared_ptr<Image> dimage = frames[0].depth_image;
    thrust::device_ptr<float> d_data(dimage->GetData());
    thrust::host_vector<float> data(d_data, d_data + dimage->GetTotal());
    cv::Mat image(h, w, CV_32FC1, data.data());
    image.convertTo(image, CV_16UC1, 10000);
    cv::imwrite("depth.png", image);
  }

  {
    std::shared_ptr<ColorImage> cimage = frames[0].color_image;
    thrust::device_ptr<Vector3f> d_data(cimage->GetData());
    thrust::host_vector<Vector3f> data(d_data, d_data + cimage->GetTotal());
    cv::Mat image(h, w, CV_32FC3, data.data());
    image.convertTo(image, CV_8UC3, 255);
    cv::cvtColor(image, image, CV_BGR2RGB);
    cv::imwrite("color.png", image);
  }

  {
    std::shared_ptr<ColorImage> nimage = frames[0].normal_image;
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