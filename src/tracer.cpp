#include <vulcan/tracer.h>
#include <vulcan/exception.h>
#include <vulcan/volume.h>

namespace vulcan
{

Tracer::Tracer(std::shared_ptr<Volume> volume) :
  volume_(volume)
{
  Initialize();
}

std::shared_ptr<Volume> Tracer::GetVolume() const
{
  return volume_;
}

void Tracer::Trace(Frame& frame)
{
  ComputePatches();
  ComputeBounds();
  ResizeFrame(frame);
  ComputePoints(frame);
  ComputeNormals(frame);
}

void Tracer::ComputePatches()
{
  // get visibility list from volume


  // for each block in visibility list

    // compute bounding box of block (in frame space)

    // compute max and min depth of bounding box

    // break frame space box into segments (16x16 pixel patches)

    // compute total number of patches needed

    // perform prefix sum on patch counts to compute offsets

    // store patch index, min depth, and max depth at offsets
    // store patch position+size, min depth, and max depth at offsets

  VULCAN_THROW("not implemented");
}

void Tracer::ComputeBounds()
{
  // for each pixel in each patch (assuming 16x16 pixel patch)

    // atomic assign min and max depths to pixel in frame

  VULCAN_THROW("not implemented");
}

void Tracer::ResizeFrame(Frame& frame)
{
  // compute size from frame projection

  // depth of frame from internals settings (color, depth, normals, etc)

  // resize frame as indicated

  VULCAN_THROW("not implemented");
}

void Tracer::ComputePoints(Frame& frame)
{
  // for each pixel in output frame

    // compute ray from camera and pixel

    // start ray at min depth found in bounds frame

    // initialize value to be "no depth"

    // while depth < max depth

      // sample SDF at current position

      // if SDF value is negative

        // break

      // compute step size

      // update ray position

    // if surface intersected (depth < max depth)

      // refine position estimate

      // sample color if needed

      // store values in frame

  VULCAN_THROW("not implemented");
}

void Tracer::ComputeNormals(Frame& frame)
{
  // for each pixel in traced frame

    // sample neighboring pixels

    // compute normal

    // store in image

  VULCAN_THROW("not implemented");
}

void Tracer::Initialize()
{
  // allocate bounds frame

  VULCAN_THROW("not implemented");
}

} // namespace vulcan