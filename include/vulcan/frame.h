#pragma once

#include <memory>
#include <vulcan/image.h>
#include <vulcan/projection.h>
#include <vulcan/transform.h>

namespace vulcan
{

struct Frame
{
  Transform transform;

  Projection projection;

  std::shared_ptr<Image> depth_image;
};

} // namespace vulcan