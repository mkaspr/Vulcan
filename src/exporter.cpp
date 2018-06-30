#include <vulcan/exporter.h>
#include <vulcan/exception.h>

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
  VULCAN_THROW("not implemented");
}

} // namespace vulcan