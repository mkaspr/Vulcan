#include <vulcan/tracer.h>
#include <vulcan/tracer.cuh>
#include <cfloat>
#include <vulcan/hash.h>
#include <vulcan/projection.h>
#include <vulcan/transform.h>
#include <vulcan/util.cuh>
#include <vulcan/voxel.h>

namespace vulcan
{

template <int BLOCK_SIZE>
VULCAN_GLOBAL
void ComputePatchesKernel(const int* indices, const HashEntry* entries,
    const Transform Tcw, const Projection projection, float block_length,
    int block_count, int image_width, int image_height, int bounds_width,
    int bounds_height, Patch* patches, int* patch_count)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  Vector2s bmax(-1, -1);
  Vector2s bmin(bounds_width, bounds_height);
  Vector2f depth_bounds(+FLT_MAX, -FLT_MAX);
  const int patch_size = Patch::max_size;

  if (index < block_count)
  {
    const int entry_index = indices[index];
    const HashEntry entry = entries[entry_index];
    const Block& block = entry.block;
    const Vector3s& origin = block.GetOrigin();
    Vector4f Xwp(0, 0, 0, 1);

    for (int z = 0; z <= 1; ++z)
    {
      Xwp[2] = block_length * (z + origin[2]);

      for (int y = 0; y <= 1; ++y)
      {
        Xwp[1] = block_length * (y + origin[1]);

        for (int x = 0; x <= 1; ++x)
        {
          Xwp[0] = block_length * (x + origin[0]);

          const Vector3f Xcp = Vector3f(Tcw * Xwp);
          Vector2f uv = projection.Project(Xcp);

          uv[0] = bounds_width * uv[0] / image_width;
          uv[1] = bounds_height * uv[1] / image_height;

          bmin[0] = clamp<short>(min((short)floorf(uv[0]), bmin[0]), 0, bounds_width - 1);
          bmin[1] = clamp<short>(min((short)floorf(uv[1]), bmin[1]), 0, bounds_height - 1);

          bmax[0] = clamp<short>(max((short)ceilf(uv[0]), bmax[0]), 0, bounds_width - 1);
          bmax[1] = clamp<short>(max((short)ceilf(uv[1]), bmax[1]), 0, bounds_height - 1);

          depth_bounds[0] = min(Xcp[2], depth_bounds[0]);
          depth_bounds[1] = max(Xcp[2], depth_bounds[1]);
        }
      }
    }
  }

  const int rx = max(0, bmax[0] - bmin[0]);
  const int ry = max(0, bmax[1] - bmin[1]);
  const int gx = (rx + patch_size - 1) / patch_size;
  const int gy = (ry + patch_size - 1) / patch_size;
  const int count = gx * gy;

  const int offset = PrefixSum<BLOCK_SIZE>(count, threadIdx.x, *patch_count);

  for (int i = 0; i < gy; ++i)
  {
    for (int j = 0; j < gx; ++j)
    {
      Patch patch;
      const int output = offset + i * gx + j;
      patch.origin = bmin + patch_size * Vector2s(j, i);
      patch.size[0] = min(patch_size, bmax[0] - patch.origin[0] + 1);
      patch.size[1] = min(patch_size, bmax[1] - patch.origin[1] + 1);
      patch.bounds = depth_bounds;
      patches[output] = patch;
    }
  }
}

VULCAN_GLOBAL
void ComputeBoundsKernel(const Patch* patches, Vector2f* bounds,
    int bounds_width, int patch_count)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < patch_count)
  {
    const Patch patch = patches[index];

    for (int i = 0; i < patch.size[1]; ++i)
    {
      const int y = patch.origin[1] + i;

      for (int j = 0; j < patch.size[0]; ++j)
      {
        const int x = patch.origin[0] + j;
        const int pixel = y * bounds_width + x;
        atomicMin(&bounds[pixel][0], patch.bounds[0]);
        atomicMax(&bounds[pixel][1], patch.bounds[1]);
      }
    }
  }
}

VULCAN_GLOBAL
void ComputePointsKernel(const HashEntry* entries, const Voxel* voxels,
    const Vector2f* bounds, int block_count, float block_length,
    float voxel_length, float trunc_length, const Transform Twc,
    const Projection projection, float* depths, Vector3f* colors,
    int image_width, int image_height, int bounds_width, int bounds_height)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < image_width && y < image_height)
  {
    const int px = bounds_width * x / image_width;
    const int py = bounds_height * y / image_height;
    const Vector2f bound = bounds[py * bounds_width + px];

    float depth = 0;
    float final_depth = 0;
    Vector3f color = Vector3f(0, 0, 0);

    if (bound[0] < bound[1])
    {
      const Vector2f uv(x + 0.5f, y + 0.5f);
      const Vector3f Xcp = projection.Unproject(uv) * bound[0];
      const Vector3f Xwp = Vector3f(Twc * Vector4f(Xcp, 1.0f));
      const Vector3f dir = Vector3f(Twc * Vector4f(Xcp, 0.0f)).Normalized();

      const uint32_t P1 = 73856093;
      const uint32_t P2 = 19349669;
      const uint32_t P3 = 83492791;
      const uint32_t K  = block_count;

      Vector3f p = Xwp;
      const Transform Tcw = Twc.Inverse();
      const int r = Block::resolution;
      const int rr = r * r;

      depth = bound[0];
      color = Vector3f(0, 0, 0);

      int iters = 0;

      do
      {
        const int bx = floorf(p[0] / block_length);
        const int by = floorf(p[1] / block_length);
        const int bz = floorf(p[2] / block_length);
        const uint32_t hash_code = ((bx * P1) ^ (by * P2) ^ (bz * P3)) % K;
        HashEntry entry = entries[hash_code];

        while (entry.block != Block(bx, by, bz) && entry.HasNext())
        {
          entry = entries[entry.next];
        }

        if (entry.block == Block(bx, by, bz) && entry.IsAllocated())
        {
          const int vx = (p[0] - bx * block_length) / voxel_length;
          const int vy = (p[1] - by * block_length) / voxel_length;
          const int vz = (p[2] - bz * block_length) / voxel_length;
          const int block_offset = Block::voxel_count * entry.data;
          const int voxel_offset = vz * rr + vy * r + vx;
          const Voxel voxel = voxels[block_offset + voxel_offset];
          float sdf = voxel.distance;

          if (sdf <= 0.1f && sdf >= -0.5f)
          {
            // compute depth via trilinear interpolation
            sdf = sdf; // TODO
          }

          if (sdf <= 0.0f) // TODO: uncomment
          {
            p += trunc_length * sdf * dir;
            // sample depth via trilinear interpolation at final point (again)
            // sample color via trilinear interpolation at final point
            const Vector3f Xcd = Vector3f(Tcw * Vector4f(p, 1.0f));
            color = voxel.color; // TODO
            final_depth = Xcd[2]; // TODO
            break;
          }
          else
          {
            p += max(voxel_length, trunc_length * sdf) * dir;
          }
        }
        else
        {
          p += block_length * dir;
        }

        const Vector3f Xcd = Vector3f(Tcw * Vector4f(p, 1.0f));
        depth = Xcd[2];

        if (++iters >= 20) // TODO: remove
        {
          // printf("iteration reached\n");
          color = Vector3f(1, 0, 0);
          break;
        }
        // else if (depth > bound[1])
        // {
        //   printf("depth reached\n");
        //   color = Vector3f(1, 1, 0);
        //   break;
        // }
      }
      while (depth < bound[1]);
    }

    const int pixel = y * image_width + x;
    depths[pixel] = final_depth;
    colors[pixel] = color;
  }
}

template <int PATCH_SIZE>
VULCAN_GLOBAL
void ComputeNormalsKernel(const float* depths, const Projection projection,
    Vector3f* normals, int image_width, int image_height)
{
  const int pad = 1;
  const int resolution = (PATCH_SIZE + 2 * pad);
  const int shared_size = resolution * resolution;
  VULCAN_SHARED float shared[shared_size];

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  int shared_index = threadIdx.y * blockDim.x + threadIdx.x;

  while (shared_index < shared_size)
  {
    float depth = 0;
    const int bx = blockIdx.x * blockDim.x - pad + (shared_index % resolution);
    const int by = blockIdx.y * blockDim.y - pad + (shared_index / resolution);

    if (bx >= 0 && bx < image_width && by >= 0 && by < image_height)
    {
      depth = depths[by * image_width + bx];
    }

    shared[shared_index] = depth;
    shared_index += blockDim.x * blockDim.y;
  }

  __syncthreads();

  const float depth = shared[(threadIdx.y + pad) * resolution + (threadIdx.x + pad)];

  if (x < image_width && y < image_height)
  {
    Vector3f normal(0, 0, 0);

    if (depth > 0)
    {
      float d;
      Vector2f uv;

      uv[0] = (x - pad) + 0.5f;
      uv[1] = (y + 0) + 0.5f;
      d = shared[(threadIdx.y + pad) * resolution + (threadIdx.x + 0)];
      const Vector3f x0 = projection.Unproject(uv) * d;

      uv[0] = (x + pad) + 0.5f;
      uv[1] = (y + 0) + 0.5f;
      d = shared[(threadIdx.y + pad) * resolution + (threadIdx.x + 2 * pad)];
      const Vector3f x1 = projection.Unproject(uv) * d;

      uv[0] = (x + 0) + 0.5f;
      uv[1] = (y - pad) + 0.5f;
      d = shared[(threadIdx.y + 0) * resolution + (threadIdx.x + pad)];
      const Vector3f y0 = projection.Unproject(uv) * d;

      uv[0] = (x + 0) + 0.5f;
      uv[1] = (y + pad) + 0.5f;
      d = shared[(threadIdx.y + 2 * pad) * resolution + (threadIdx.x + pad)];
      const Vector3f y1 = projection.Unproject(uv) * d;

      const Vector3f dx = x0 - x1;
      const Vector3f dy = y0 - y1;

      normal = dx.Cross(dy);
      normal.Normalize();
    }

    const int output = y * image_width + x;
    normals[output] = normal.Normalized();
  }
}

void ComputePatches(const int* indices, const HashEntry* entries,
    const Transform& Tcw, const Projection& projection, float block_length,
    int block_count, int image_width, int image_height, int bounds_width,
    int bounds_height, Patch* patches, int* patch_count)
{
  const int threads = 512;
  const int blocks = GetKernelBlocks(block_count, threads);

  CUDA_LAUNCH(ComputePatchesKernel<threads>, blocks, threads, 0, 0,
      indices, entries, Tcw, projection, block_length, block_count,
      image_width, image_height, bounds_width, bounds_height, patches,
      patch_count);
}

void ComputeBounds(const Patch* patches, Vector2f* bounds, int bounds_width,
    int patch_count)
{
  const int threads = 512;
  const int blocks = GetKernelBlocks(patch_count, threads);

  CUDA_LAUNCH(ComputeBoundsKernel, blocks, threads, 0, 0, patches,
      bounds, bounds_width, patch_count);
}

void ComputePoints(const HashEntry* entries, const Voxel* voxels,
    const Vector2f* bounds, int block_count, float block_length,
    float voxel_length, float trunc_length, const Transform& Twc,
    const Projection& projection, float* depths, Vector3f* colors,
    int image_width, int image_height, int bounds_width, int bounds_height)
{
  const dim3 threads(16, 16);
  const dim3 total(image_width, image_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(ComputePointsKernel, blocks, threads, 0, 0, entries, voxels,
      bounds, block_count, block_length, voxel_length, trunc_length, Twc,
      projection, depths, colors, image_width, image_height,
      bounds_width, bounds_height);
}

void ComputeNormals(const float* depths, const Projection& projection,
    Vector3f* normals, int image_width, int image_height)
{
  const dim3 threads(16, 16);
  const dim3 total(image_width, image_height);
  const dim3 blocks = GetKernelBlocks(total, threads);

  CUDA_LAUNCH(ComputeNormalsKernel<16>, blocks, threads, 0, 0, depths,
      projection, normals, image_width, image_height);
}

void ResetBoundsBuffer(Vector2f *bounds, int count)
{
  const int threads = 512;
  const int blocks = GetKernelBlocks(count, threads);
  const Vector2f value(+FLT_MAX, -FLT_MAX);
  CUDA_LAUNCH(FillKernel, blocks, threads, 0, 0, bounds, value, count);
}

} // namespace vulcan