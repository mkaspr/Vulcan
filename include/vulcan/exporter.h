#pragma once

#include <string>

namespace vulcan
{

class Mesh;

class Exporter
{
  public:

    Exporter(const std::string& file);

    const std::string& GetFile() const;

    void Export(const Mesh& mesh) const;

  protected:

    std::string file_;
};

} // namespace vulcan