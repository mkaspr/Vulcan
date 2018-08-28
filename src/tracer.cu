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
    float min_depth, float max_depth, int block_count, int image_width,
    int image_height, int bounds_width, int bounds_height, Patch* patches,
    int* patch_count)
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

          depth_bounds[0] = clamp(min(Xcp[2], depth_bounds[0]), min_depth, max_depth);
          depth_bounds[1] = clamp(max(Xcp[2], depth_bounds[1]), min_depth, max_depth);
        }
      }
    }
  }

  const int rx = bmax[0] - bmin[0];
  const int ry = bmax[1] - bmin[1];
  const int gx = (rx + patch_size - 1) / patch_size;
  const int gy = (ry + patch_size - 1) / patch_size;
  const int count = (depth_bounds[1] > depth_bounds[0]) ? gx * gy : 0;
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

VULCAN_DEVICE
Voxel GetVoxel(int K, const HashEntry* entries, const Voxel* voxels,
    int bx, int by, int bz, int vx, int vy, int vz)
{
  if (vx < 0)
  {
    --bx;
    vx = Block::resolution + vx;
  }
  else if (vx >= Block::resolution)
  {
    ++bx;
    vx = vx - Block::resolution;
  }

  if (vy < 0)
  {
    --by;
    vy = Block::resolution + vy;
  }
  else if (vy >= Block::resolution)
  {
    ++by;
    vy = vy - Block::resolution;
  }

  if (vz < 0)
  {
    --bz;
    vz = Block::resolution + vz;
  }
  else if (vz >= Block::resolution)
  {
    ++bz;
    vz = vz - Block::resolution;
  }

  const uint32_t P1 = 73856093;
  const uint32_t P2 = 19349669;
  const uint32_t P3 = 83492791;

  const uint32_t hash_code = ((bx * P1) ^ (by * P2) ^ (bz * P3)) % K;
  HashEntry entry = entries[hash_code];
  bool found = false;

  do
  {
    if (entry.block == Block(bx, by, bz))
    {
      found = true;
      break;
    }
    else if (!entry.HasNext())
    {
      break;
    }

    entry = entries[entry.next];
  }
  while (true);

  if (found && entry.IsAllocated())
  {
    const int r = Block::resolution;
    const int rr = r * r;

    const int block_offset = Block::voxel_count * entry.data;
    const int voxel_offset = vz * rr + vy * r + vx;
    return voxels[block_offset + voxel_offset];
  }
  else
  {
    return Voxel::Empty();
  }
}

VULCAN_DEVICE
Vector4f GetInterpolatedDistance(const HashEntry* entries, const Voxel* voxels,
    int K, float block_length, float voxel_length, int bx, int by, int bz,
    const HashEntry& entry, const Vector3f& p, float sdf)
{
  const float wx = (p[0] - bx * block_length) / voxel_length;
  const float wy = (p[1] - by * block_length) / voxel_length;
  const float wz = (p[2] - bz * block_length) / voxel_length;

  const int r = Block::resolution;
  const int rr = r * r;

  const int block_offset = Block::voxel_count * entry.data;

  Vector3f color;

  const int i0x = floorf(wx - 0.5f);
  const int i0y = floorf(wy - 0.5f);
  const int i0z = floorf(wz - 0.5f);

  Voxel v000;
  Voxel v001;
  Voxel v010;
  Voxel v011;
  Voxel v100;
  Voxel v101;
  Voxel v110;
  Voxel v111;

  if (i0x >= 0 && i0y >= 0 && i0z >= 0 &&
      i0x < Block::resolution - 1 &&
      i0y < Block::resolution - 1 &&
      i0z < Block::resolution - 1)
  {
    // all samples come from current block

    const int i000 = (i0z + 0) * rr + (i0y + 0) * r + (i0x + 0);
    const int i001 = (i0z + 0) * rr + (i0y + 0) * r + (i0x + 1);
    const int i010 = (i0z + 0) * rr + (i0y + 1) * r + (i0x + 0);
    const int i011 = (i0z + 0) * rr + (i0y + 1) * r + (i0x + 1);
    const int i100 = (i0z + 1) * rr + (i0y + 0) * r + (i0x + 0);
    const int i101 = (i0z + 1) * rr + (i0y + 0) * r + (i0x + 1);
    const int i110 = (i0z + 1) * rr + (i0y + 1) * r + (i0x + 0);
    const int i111 = (i0z + 1) * rr + (i0y + 1) * r + (i0x + 1);

    v000 = voxels[block_offset + i000];
    v001 = voxels[block_offset + i001];
    v010 = voxels[block_offset + i010];
    v011 = voxels[block_offset + i011];
    v100 = voxels[block_offset + i100];
    v101 = voxels[block_offset + i101];
    v110 = voxels[block_offset + i110];
    v111 = voxels[block_offset + i111];
  }
  else
  {
    // samples come from multiple block

    v000 = GetVoxel(K, entries, voxels, bx, by, bz, i0x + 0, i0y + 0, i0z + 0);
    v001 = GetVoxel(K, entries, voxels, bx, by, bz, i0x + 1, i0y + 0, i0z + 0);
    v010 = GetVoxel(K, entries, voxels, bx, by, bz, i0x + 0, i0y + 1, i0z + 0);
    v011 = GetVoxel(K, entries, voxels, bx, by, bz, i0x + 1, i0y + 1, i0z + 0);
    v100 = GetVoxel(K, entries, voxels, bx, by, bz, i0x + 0, i0y + 0, i0z + 1);
    v101 = GetVoxel(K, entries, voxels, bx, by, bz, i0x + 1, i0y + 0, i0z + 1);
    v110 = GetVoxel(K, entries, voxels, bx, by, bz, i0x + 0, i0y + 1, i0z + 1);
    v111 = GetVoxel(K, entries, voxels, bx, by, bz, i0x + 1, i0y + 1, i0z + 1);
  }

  Vector3f w1;
  w1[0] = wx - (i0x + 0.5f);
  w1[1] = wy - (i0y + 0.5f);
  w1[2] = wz - (i0z + 0.5f);

  Vector3f w0 = Vector3f::Ones() - w1;

  const float n000 = v000.distance;
  const float n001 = v001.distance;
  const float n010 = v010.distance;
  const float n011 = v011.distance;
  const float n100 = v100.distance;
  const float n101 = v101.distance;
  const float n110 = v110.distance;
  const float n111 = v111.distance;

  const float n00 = n000 * w0[0] + n001 * w1[0];
  const float n01 = n010 * w0[0] + n011 * w1[0];
  const float n10 = n100 * w0[0] + n101 * w1[0];
  const float n11 = n110 * w0[0] + n111 * w1[0];

  const float n0 = n00 * w0[1] + n01 * w1[1];
  const float n1 = n10 * w0[1] + n11 * w1[1];

  const float w000 = w0[2] * w0[1] * w0[0] * (v000.color_weight > 0) ? 1 : 0;
  const float w001 = w0[2] * w0[1] * w1[0] * (v001.color_weight > 0) ? 1 : 0;
  const float w010 = w0[2] * w1[1] * w0[0] * (v010.color_weight > 0) ? 1 : 0;
  const float w011 = w0[2] * w1[1] * w1[0] * (v011.color_weight > 0) ? 1 : 0;
  const float w100 = w1[2] * w0[1] * w0[0] * (v100.color_weight > 0) ? 1 : 0;
  const float w101 = w1[2] * w0[1] * w1[0] * (v101.color_weight > 0) ? 1 : 0;
  const float w110 = w1[2] * w1[1] * w0[0] * (v110.color_weight > 0) ? 1 : 0;
  const float w111 = w1[2] * w1[1] * w1[0] * (v111.color_weight > 0) ? 1 : 0;

  float total = 0;
  total += w000;
  total += w001;
  total += w010;
  total += w011;
  total += w100;
  total += w101;
  total += w110;
  total += w111;

  color = Vector3f(0, 0, 0);
  color += w000 * v000.GetColor();
  color += w001 * v001.GetColor();
  color += w010 * v010.GetColor();
  color += w011 * v011.GetColor();
  color += w100 * v100.GetColor();
  color += w101 * v101.GetColor();
  color += w110 * v110.GetColor();
  color += w111 * v111.GetColor();
  if (total > 0) color /= total;

  sdf = n0 * w0[2] + n1 * w1[2];

  return Vector4f(sdf, color[0], color[1], color[2]);
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
      const Vector3f Xcp = projection.Unproject(uv, bound[0]);
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
          const float wx = (p[0] - bx * block_length) / voxel_length;
          const float wy = (p[1] - by * block_length) / voxel_length;
          const float wz = (p[2] - bz * block_length) / voxel_length;

          const int vx = wx;
          const int vy = wy;
          const int vz = wz;

          const int block_offset = Block::voxel_count * entry.data;
          const int voxel_offset = vz * rr + vy * r + vx;
          const Voxel voxel = voxels[block_offset + voxel_offset];
          float sdf = voxel.distance;

          if (sdf <= 0.1f && sdf >= -0.5f)
          {
            const Vector4f v = GetInterpolatedDistance(entries, voxels, K,
                block_length, voxel_length, bx, by, bz, entry, p, sdf);

            sdf = v[0];
            color = Vector3f(v[1], v[2], v[3]);
          }

          if (sdf <= 0.0f)
          {
            p += trunc_length * sdf * dir;

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
              const Vector4f v = GetInterpolatedDistance(entries, voxels, K,
                  block_length, voxel_length, bx, by, bz, entry, p, sdf);

              sdf = v[0];
              color = Vector3f(v[1], v[2], v[3]);
              p += trunc_length * sdf * dir;
            }

            const Vector3f Xcd = Vector3f(Tcw * Vector4f(p, 1.0f));
            final_depth = Xcd[2];
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

        if (++iters >= 200) // TODO: remove
        {
          // printf("iteration reached\n");
          color = Vector3f(1, 0, 0);
          break;
        }
      }
      while (depth < bound[1]);
    }

    const int pixel = y * image_width + x;
    depths[pixel] = final_depth;
    colors[pixel] = color;
  }
}

void ComputePatches(const int* indices, const HashEntry* entries,
    const Transform& Tcw, const Projection& projection, float block_length,
    float min_depth, float max_depth, int block_count, int image_width,
    int image_height, int bounds_width, int bounds_height, Patch* patches,
    int* patch_count)
{
  const int threads = 512;
  const int blocks = GetKernelBlocks(block_count, threads);

  CUDA_LAUNCH(ComputePatchesKernel<threads>, blocks, threads, 0, 0,
      indices, entries, Tcw, projection, block_length, min_depth, max_depth,
      block_count, image_width, image_height, bounds_width, bounds_height,
      patches, patch_count);
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

void ResetBoundsBuffer(Vector2f *bounds, int count)
{
  const int threads = 512;
  const int blocks = GetKernelBlocks(count, threads);
  const Vector2f value(+FLT_MAX, -FLT_MAX);
  CUDA_LAUNCH(FillKernel, blocks, threads, 0, 0, bounds, value, count);
}

} // namespace vulcan