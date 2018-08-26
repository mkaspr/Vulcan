#pragma once

#include <memory>
#include <vulcan/image.h>
#include <vulcan/projection.h>
#include <vulcan/transform.h>

namespace vulcan
{

struct Frame
{
  Transform Twc;

  Projection projection;

  // TODO: add separate color transform

  // TODO: add separate color projection

  std::shared_ptr<Image> depth_image;

  std::shared_ptr<ColorImage> color_image;

  std::shared_ptr<ColorImage> normal_image;

  void ComputeNormals();

  void Downsample(Frame& frame);
};

} // namespace vulcan