#include <vulcan/frame.h>
#include <vulcan/frame.cuh>
#include <vulcan/exception.h>

namespace vulcan
{

void Frame::FilterDepths()
{
  VULCAN_ASSERT_MSG(depth_image, "missing depth image");

  const int w = depth_image->GetWidth();
  const int h = depth_image->GetHeight();
  Image temp(depth_image->GetWidth(), depth_image->GetHeight());
  float* src = depth_image->GetData();
  float* dst = temp.GetData();
  vulcan::FilterDepths(w, h, src, dst);
  CUDA_DEBUG(cudaMemcpy(src, dst, temp.GetBytes(), cudaMemcpyDeviceToDevice));
}

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
  vulcan::ComputeNormals(depths, depth_projection, normals, w, h);
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

  frame.depth_projection.SetFocalLength(depth_projection.GetFocalLength() / 2);
  frame.depth_projection.SetCenterPoint(depth_projection.GetCenterPoint() / 2);
  frame.color_projection.SetFocalLength(color_projection.GetFocalLength() / 2);
  frame.color_projection.SetCenterPoint(color_projection.GetCenterPoint() / 2);
  frame.depth_to_world_transform = depth_to_world_transform;
  frame.depth_to_color_transform = depth_to_color_transform;
}


} // namespace vulcan