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

} // namespace vulcan