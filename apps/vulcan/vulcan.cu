#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <vulcan/vulcan.h>

#include <ctime>

DEFINE_int32(main_blocks, 260096, "main block count");
DEFINE_int32(excess_blocks, 2048, "excess block count");
DEFINE_double(voxel_length, 0.02, "voxel edge length");
DEFINE_double(truncation_length, 0.1, "volume truncation length");
DEFINE_string(output, "output.ply", "output mesh file");

using namespace vulcan;

clock_t start;

inline void Tic()
{
  start = clock();
}

inline void Toc(const std::string& name)
{

  const clock_t stop = clock();
  const double time = double(stop - start) / CLOCKS_PER_SEC;
  LOG(INFO) << name << " time: " << time / 100;
}

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "Reading frames...";

  Tic();

  std::shared_ptr<Image> image = std::make_shared<Image>();
  const std::string folder = "/home/mike/Datasets/Work/kitchen02/images";
  const std::string file = folder + "/depth_0100.png";
  image->Load(file, 1.0 / 1000.0);

  std::vector<Frame> frames(1);
  frames[0].transform = Transform::Translate(0, 0, 0);
  frames[0].projection.SetFocalLength(567.3940, 567.4752);
  frames[0].projection.SetCenterPoint(319.9336, 240.4598);
  frames[0].depth_image = image;

  Toc("Reading");

  LOG(INFO) << "Creating volume...";

  Tic();

  std::shared_ptr<Volume> volume;
  volume = std::make_shared<Volume>(FLAGS_main_blocks, FLAGS_excess_blocks);
  volume->SetTruncationLength(FLAGS_truncation_length);
  volume->SetVoxelLength(FLAGS_voxel_length);

  Toc("Creating volume");

  LOG(INFO) << "Creating integrator...";

  Tic();

  Integrator integrator(volume);

  Toc("Creating integrator");

  LOG(INFO) << "Integrating frames...";

  Tic();

  // for (const Frame& frame : frames)
  for (int i = 0; i < 100; ++i)
  {
    const Frame& frame = frames[0];
    volume->SetView(frame);
    integrator.Integrate(frame);
    CUDA_ASSERT(cudaDeviceSynchronize());
    LOG(INFO) << "Visible blocks: " << volume->GetVisibleBlocks().GetSize();
  }

  Toc("Integrating frames");

  // LOG(INFO) << "Extracting mesh...";

  // Mesh mesh;
  // Extractor extractor(volume);
  // extractor.Extract(mesh);

  // LOG(INFO) << "Writing mesh...";

  // Exporter exporter(FLAGS_output);
  // exporter.Export(mesh);

  LOG(INFO) << "Success";
  return 0;
}