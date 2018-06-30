#pragma once

#include <vulcan/buffer.h>
#include <vulcan/matrix.h>

namespace vulcan
{

struct Mesh
{
  Buffer<Vector3f> points;

  Buffer<Vector3f> normals;

  Buffer<Vector3i> faces;
};

} // namespace vulcan