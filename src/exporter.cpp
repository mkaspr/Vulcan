#include <vulcan/exporter.h>
#include <fstream>
#include <vulcan/exception.h>
#include <vulcan/mesh.h>

namespace vulcan
{

Exporter::Exporter(const std::string& file) :
  file_(file)
{
}

const std::string& Exporter::GetFile() const
{
  return file_;
}

void Exporter::Export(const Mesh& mesh) const
{
  std::ofstream fout(file_);
  VULCAN_ASSERT(fout.is_open());

  fout << "ply" << std::endl;
  fout << "format ascii 1.0" << std::endl;
  fout << "element vertex " << mesh.points.size() << std::endl;
  fout << "property float x" << std::endl;
  fout << "property float y" << std::endl;
  fout << "property float z" << std::endl;
  fout << "property uchar red" << std::endl; // REMOVE
  fout << "property uchar green" << std::endl; // REMOVE
  fout << "property uchar blue" << std::endl; // REMOVE
  fout << "element face " << mesh.faces.size() << std::endl;
  fout << "property list uchar int vertex_indices" << std::endl;
  fout << "end_header" << std::endl;

  int index = 0;
  float dmin, dmax;

  for (const Vector3f& point : mesh.points)
  {
    if (index == 0 || point[2] < dmin)
    {
      dmin = point[2];
    }

    if (index == 0 || point[2] > dmax)
    {
      dmax = point[2];
    }

    ++index;
  }

  dmin = 0.35f;

  for (const Vector3f& point : mesh.points)
  {
    // fout << point[0] << " " << point[1] << " " << point[2] << std::endl;
    fout << point[0] << " " << point[1] << " " << point[2] << " ";
    const int color = int(255 * min(1.0f, (point[2] - dmin) / (dmax - dmin)));
    fout << color << " " << color << " " << color << std::endl;
  }

  for (const Vector3i& face : mesh.faces)
  {
    fout << "3 " << face[0] << " " << face[1] << " " << face[2] << std::endl;
  }

  fout.close();
}

} // namespace vulcan