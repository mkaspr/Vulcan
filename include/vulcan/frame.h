#pragma once

#include <memory>
#include <vulcan/image.h>
#include <vulcan/projection.h>
#include <vulcan/transform.h>

namespace vulcan
{

struct Frame
{
  Projection depth_projection;

  Projection color_projection;

  Transform depth_to_world_transform;

  Transform depth_to_color_transform;

  std::shared_ptr<Image> depth_image;

  std::shared_ptr<ColorImage> color_image;

  std::shared_ptr<ColorImage> normal_image;

  void FilterDepths();

  void ComputeNormals();

  void Downsample(Frame& frame) const;
};

} // namespace vulcan