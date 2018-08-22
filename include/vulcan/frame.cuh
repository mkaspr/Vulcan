#pragma once

#include <vulcan/matrix.h>

namespace vulcan
{

class Projection;

void ComputeNormals(const float* depths, const Projection& projection,
    Vector3f* normals, int image_width, int image_height);

} // namespace vulcan