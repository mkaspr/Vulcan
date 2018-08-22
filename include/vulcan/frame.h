#pragma once

#include <memory>
#include <vulcan/image.h>
#include <vulcan/projection.h>
#include <vulcan/transform.h>

namespace vulcan
{

struct Frame
{
  Transform Tcw;

  Projection projection;

  // TODO: add separate color transform

  // TODO: add separate color projection

  // TODO: add light model (transform, intensity, etc)

  std::shared_ptr<Image> depth_image;

  std::shared_ptr<ColorImage> color_image;

  std::shared_ptr<ColorImage> normal_image;

  void ComputeNormals();
};

} // namespace vulcan