#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <vulcan/vulcan.h>

DEFINE_int32(main_blocks, 260096, "main block count");
DEFINE_int32(excess_blocks, 2048, "excess block count");
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

  std::vector<Frame> frames(2);
  std::shared_ptr<Image> image;
  image = std::make_shared<Image>();

  // {
  //   const std::string folder = "/home/mike/Documents/Matlab/Marble/";
  //   const std::string file = folder + "/depth_sphere.png";
  //   image->Load(file, 1.0 / 1000.0);

  //   frames[0].transform = Transform::Translate(0, 0, 0);
  //   // frames[0].projection.SetFocalLength(567.3940, 567.4752);
  //   // frames[0].projection.SetCenterPoint(319.9336, 240.4598);
  //   frames[0].projection.SetFocalLength(567, 567);
  //   frames[0].projection.SetCenterPoint(320, 240);
  //   frames[0].depth_image = image;
  // }

  {
    // const std::string folder = "/home/mike/Datasets/Work/kitchen02/images";
    const std::string folder = "/home/mike/Datasets/Work/volley02/images";
    const std::string file = folder + "/depth_0000.png";
    image->Load(file, 1.0 / 1000.0);

    frames[0].transform = Transform::Translate(0, 0, 0);
    frames[0].projection.SetFocalLength(567.3940, 567.4752);
    frames[0].projection.SetCenterPoint(319.9336, 240.4598);
    frames[0].depth_image = image;
  }

  {
    const std::string folder = "/home/mike/Datasets/Work/volley02/images";
    const std::string file = folder + "/depth_0001.png";
    image->Load(file, 1.0 / 1000.0);

    frames[1].transform = Transform::Translate(0, 0, 0);
    frames[1].projection.SetFocalLength(567.3940, 567.4752);
    frames[1].projection.SetCenterPoint(319.9336, 240.4598);
    frames[1].depth_image = image;
  }

  // {
  //   const std::string folder = "/home/mike/Datasets/Work/kitchen02/images";
  //   const std::string file = folder + "/depth_0102.png";
  //   image->Load(file, 1.0 / 1000.0);

  //   frames[2].transform = Transform::Translate(0, 0, 0);
  //   frames[2].projection.SetFocalLength(567.3940, 567.4752);
  //   frames[2].projection.SetCenterPoint(319.9336, 240.4598);
  //   frames[2].depth_image = image;
  // }

  // {
  //   const std::string folder = "/home/mike/Datasets/Work/kitchen02/images";
  //   const std::string file = folder + "/depth_0103.png";
  //   image->Load(file, 1.0 / 1000.0);

  //   frames[3].transform = Transform::Translate(0, 0, 0);
  //   frames[3].projection.SetFocalLength(567.3940, 567.4752);
  //   frames[3].projection.SetCenterPoint(319.9336, 240.4598);
  //   frames[3].depth_image = image;
  // }

  // {
  //   const std::string folder = "/home/mike/Datasets/Work/kitchen02/images";
  //   const std::string file = folder + "/depth_0104.png";
  //   image->Load(file, 1.0 / 1000.0);

  //   frames[4].transform = Transform::Translate(0, 0, 0);
  //   frames[4].projection.SetFocalLength(567.3940, 567.4752);
  //   frames[4].projection.SetCenterPoint(319.9336, 240.4598);
  //   frames[4].depth_image = image;
  // }

  // {
  //   const std::string folder = "/home/mike/Datasets/Work/kitchen02/images";
  //   const std::string file = folder + "/depth_0105.png";
  //   image->Load(file, 1.0 / 1000.0);

  //   frames[5].transform = Transform::Translate(0, 0, 0);
  //   frames[5].projection.SetFocalLength(567.3940, 567.4752);
  //   frames[5].projection.SetCenterPoint(319.9336, 240.4598);
  //   frames[5].depth_image = image;
  // }

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
    CUDA_ASSERT(cudaDeviceSynchronize());
    // // REMOVE

    volume->SetView(frame);
    integrator.Integrate(frame);
    CUDA_ASSERT(cudaDeviceSynchronize());
  }

  LOG(INFO) << "Extracting mesh...";

  Mesh mesh;
  DeviceMesh dmesh;
  Extractor extractor(volume);

  const clock_t start = clock();

  extractor.Extract(mesh);
  extractor.Extract(dmesh);

  const clock_t stop = clock();
  const double time = double(stop - start) / CLOCKS_PER_SEC;
  LOG(INFO) << "Extract time: " << time << " (" << 1 / time << " fps)";

  LOG(INFO) << "Writing mesh...";

  Exporter exporter(FLAGS_output);
  exporter.Export(mesh);

  LOG(INFO) << "Success";
  return 0;
}