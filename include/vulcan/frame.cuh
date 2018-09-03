#pragma once

#include <vulcan/matrix.h>

namespace vulcan
{

class Projection;

void ComputeNormals(const float* depths, const Projection& projection,
    Vector3f* normals, int image_width, int image_height);

void FilterDepths(int image_width, int image_height, const float* src,
    float* dst);

} // namespace vulcan