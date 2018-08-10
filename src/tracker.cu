#include <vulcan/tracker.h>
#include <vulcan/projection.h>
#include <vulcan/transform.h>

namespace vulcan
{

VULCAN_DEVICE
Vector3f Sample(const Vector3f* colors, const Vector2f& uv, int w)
{
  const Vector2i base(uv[0] - 0.5f, uv[1] - 0.5f);
  const Vector2f w1 = uv - (Vector2f(base) + 0.5f);
  const Vector2f w0 = Vector2f::Ones() - w1;

  const Vector3f c00 = colors[w * (base[1] + 0) + (base[0] + 0)];
  const Vector3f c01 = colors[w * (base[1] + 0) + (base[0] + 1)];
  const Vector3f c10 = colors[w * (base[1] + 1) + (base[0] + 0)];
  const Vector3f c11 = colors[w * (base[1] + 1) + (base[0] + 1)];

  Vector3f result(0, 0, 0);
  result += w0[1] * w0[0] * c00;
  result += w0[1] * w1[0] * c01;
  result += w1[1] * w0[0] * c10;
  result += w1[1] * w1[0] * c11;

  return result;
}

VULCAN_GLOBAL
void ComputeResidualsKernel(
    const Transform Tmr,
    const float* r_depths,
    const Vector3f* r_colors,
    const Vector3f* r_normals,
    const Projection r_projection,
    const Vector3f* m_colors,
    const int r_width,
    const int r_height,
    const Projection m_projection,
    const Vector3f m_light_position,
    const float m_light_intensity,
    const int m_width,
    const int m_height,
    double* residuals)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < r_width && y < r_height)
  {
    const int r_pixel = y * r_width + x;
    const float r_depth = r_depths[r_pixel];

    if (r_depth > 0)
    {
      const Vector2f r_uv = Vector2f(x + 0.5f, y + 0.5f);
      const Vector3f Xrp = r_projection.Unproject(r_uv, r_depth);
      const Vector3f Xmp = Vector3f(Tmr * Vector4f(Xrp, 1));
      const Vector2f m_uv = m_projection.Project(Xmp);

      if (m_uv[0] >= 0.5f && m_uv[0] <= (m_width  - 0.5f) &&
          m_uv[1] >= 0.5f && m_uv[1] <= (m_height - 0.5f))
      {
        const Vector3f r_color = r_colors[r_pixel];
        const Vector3f Xrn = r_normals[r_pixel];
        const Vector3f Xmn = Vector3f(Tmr * Vector4f(Xrn, 0));
        Vector3f pl = m_light_position - Xmp;
        const float dd = pl.SquaredNorm();
        const float cos_theta = Xmn.Dot(pl.Normalized());
        const Vector3f r_int = r_color * m_light_intensity * cos_theta / dd;
        const Vector3f m_int = Sample(m_colors, m_uv, m_width);

        const int offset = 3 * y * r_width + x;
        residuals[offset + 0] = m_int[0] - r_int[0];
        residuals[offset + 1] = m_int[1] - r_int[1];
        residuals[offset + 2] = m_int[2] - r_int[2];
      }
    }
  }
}

VULCAN_GLOBAL
void ComputeJacobianKernel()
{
}

void Tracker::ComputeHessian()
{
  const double one = 1;
  const int residual_count = 640 * 480 * 3; // TODO: resize with image dims
  const double* jacobian = jacobian_.GetData();
  double* hessian = hessian_.GetData();

  CUBLAS_DEBUG(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, 6, 6,
      residual_count, &one, jacobian, 6, jacobian, 6, &one, hessian, 6));
}

void Tracker::ComputeGradient()
{
  const double one = 1;
  const int residual_count = 640 * 480 * 3; // TODO: resize with image dims
  const double* jacobian = jacobian_.GetData();
  const double* residuals = residuals_.GetData();
  double* gradient = gradient_.GetData();

  CUBLAS_DEBUG(cublasDgemv(handle_, CUBLAS_OP_N, 6, residual_count, &one,
    jacobian, 6, residuals, 1, &one, gradient, 1));
}

void Tracker::UpdateCost()
{
  ComputeResidual();
  prev_cost_ = curr_cost_;
  const int count = 640 * 480 * 3; // TODO: resize with image dims
  const double* data = residuals_.GetData();
  CUBLAS_DEBUG(cublasDdot(handle_, count, data, 1, data, 1, &curr_cost_));
}

} // namespace vulcan