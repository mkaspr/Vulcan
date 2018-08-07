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

  std::shared_ptr<Image> depth_image;

  std::shared_ptr<ColorImage> color_image;
};

} // namespace vulcan