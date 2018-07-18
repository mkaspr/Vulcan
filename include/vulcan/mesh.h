#pragma once

#include <vector>
#include <vulcan/buffer.h>
#include <vulcan/matrix.h>

namespace vulcan
{

struct Mesh
{
  std::vector<Vector3f> points;

  std::vector<Vector3i> faces;
};

struct DeviceMesh
{
  Buffer<Vector3f> points;

  Buffer<Vector3i> faces;
};

} // namespace vulcan