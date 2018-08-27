#include <vulcan/frame.h>
#include <vulcan/frame.cuh>
#include <vulcan/exception.h>

namespace vulcan
{

void Frame::ComputeNormals()
{
  VULCAN_ASSERT_MSG(depth_image, "missing depth image");

  if (!normal_image)
  {
    normal_image = std::make_shared<ColorImage>();
  }

  const int w = depth_image->GetWidth();
  const int h = depth_image->GetHeight();
  normal_image->Resize(w, h);
  const float* depths = depth_image->GetData();
  Vector3f* normals = normal_image->GetData();
  vulcan::ComputeNormals(depths, projection, normals, w, h);
}

void Frame::Downsample(Frame& frame) const
{
  VULCAN_DEBUG(depth_image);
  VULCAN_DEBUG(color_image);
  VULCAN_DEBUG(normal_image);

  if (!frame.depth_image) frame.depth_image = std::make_shared<Image>();
  if (!frame.color_image) frame.color_image = std::make_shared<ColorImage>();
  if (!frame.normal_image) frame.normal_image = std::make_shared<ColorImage>();

  depth_image->Downsample(*frame.depth_image, true);
  color_image->Downsample(*frame.color_image, false);
  normal_image->Downsample(*frame.normal_image, true);

  frame.projection.SetFocalLength(projection.GetFocalLength() / 2);
  frame.projection.SetCenterPoint(projection.GetCenterPoint() / 2);
  frame.Twc = Twc;
}

} // namespace vulcan