#pragma once

#include <vulcan/matrix.h>

namespace vulcan
{

class HashEntry;
class Patch;
class Projection;
class Transform;
class Voxel;

void ComputePatches(const int* indices, const HashEntry* entries,
    const Transform& Tcw, const Projection& projection, float block_length,
    int block_count, int image_width, int image_height, int bounds_width,
    int bounds_height, Patch* patches, int* patch_count);

void ComputeBounds(const Patch* patches, Vector2f* bounds, int bounds_width,
    int patch_count);

void ComputePoints(const HashEntry* entries, const Voxel* voxels,
    const Vector2f* bounds, int block_count, float block_length,
    float voxel_length, float trunc_length, const Transform& Twc,
    const Projection& projection, float* depths, Vector3f* colors,
    int image_width, int image_height, int bounds_width, int bounds_height);

void ResetBoundsBuffer(Vector2f* bounds, int count);

} // namespace vulcan