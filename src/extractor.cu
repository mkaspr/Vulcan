#include <vulcan/extractor.h>
#include <vulcan/block.h>
#include <vulcan/exception.h>
#include <vulcan/hash.h>
#include <vulcan/mesh.h>
#include <vulcan/util.cuh>
#include <vulcan/volume.h>
#include <vulcan/voxel.h>

#include <iostream>

namespace vulcan
{

static VULCAN_CONSTANT char edge_coords[24][4] =
{
  { 0, 0, 0, 0 }, { 1, 0, 0, 0 }, //  0
  { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, //  1
  { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, //  2
  { 0, 0, 0, 0 }, { 0, 1, 0, 0 }, //  3
  { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, //  4
  { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, //  5
  { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, //  6
  { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, //  7
  { 0, 0, 0, 0 }, { 0, 0, 1, 0 }, //  8
  { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, //  9
  { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, // 10
  { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, // 11
};

static VULCAN_CONSTANT int edge_counts[256] =
{
  0, 3, 1, 2, 1, 2, 2, 1, 0, 3, 1, 2, 1, 2, 2, 1,
  1, 2, 2, 1, 2, 1, 3, 0, 1, 2, 2, 1, 2, 1, 3, 0,
  0, 3, 1, 2, 1, 2, 2, 1, 0, 3, 1, 2, 1, 2, 2, 1,
  1, 2, 2, 1, 2, 1, 3, 0, 1, 2, 2, 1, 2, 1, 3, 0,
  0, 3, 1, 2, 1, 2, 2, 1, 0, 3, 1, 2, 1, 2, 2, 1,
  1, 2, 2, 1, 2, 1, 3, 0, 1, 2, 2, 1, 2, 1, 3, 0,
  0, 3, 1, 2, 1, 2, 2, 1, 0, 3, 1, 2, 1, 2, 2, 1,
  1, 2, 2, 1, 2, 1, 3, 0, 1, 2, 2, 1, 2, 1, 3, 0,
  0, 3, 1, 2, 1, 2, 2, 1, 0, 3, 1, 2, 1, 2, 2, 1,
  1, 2, 2, 1, 2, 1, 3, 0, 1, 2, 2, 1, 2, 1, 3, 0,
  0, 3, 1, 2, 1, 2, 2, 1, 0, 3, 1, 2, 1, 2, 2, 1,
  1, 2, 2, 1, 2, 1, 3, 0, 1, 2, 2, 1, 2, 1, 3, 0,
  0, 3, 1, 2, 1, 2, 2, 1, 0, 3, 1, 2, 1, 2, 2, 1,
  1, 2, 2, 1, 2, 1, 3, 0, 1, 2, 2, 1, 2, 1, 3, 0,
  0, 3, 1, 2, 1, 2, 2, 1, 0, 3, 1, 2, 1, 2, 2, 1,
  1, 2, 2, 1, 2, 1, 3, 0, 1, 2, 2, 1, 2, 1, 3, 0,
};

static VULCAN_CONSTANT char edge_indices[256][4] =
{
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 }, { 3, 8, 0, 0 },
  { 3, 0, 0, 0 }, { 0, 8, 0, 0 }, { 0, 3, 0, 0 }, { 8, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
  { 8, 0, 0, 0 }, { 0, 3, 0, 0 }, { 0, 8, 0, 0 }, { 3, 0, 0, 0 },
  { 3, 8, 0, 0 }, { 0, 0, 0, 0 }, { 0, 3, 8, 0 }, { 0, 0, 0, 0 },
};

// static const __constant__ int face_counts[256] =
// {
// };
//
// static const __constant__ Vector3i face_indices[256] =
// {
// };

////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_SIZE>
VULCAN_GLOBAL
void ExtractCubeStateKernel(const int* visible_blocks, int block_index,
    const HashEntry* hash_entries, const Voxel* voxels, CubeState* states,
    int* state_count)
{
  // allocate shared memory for reading in voxel data
  VULCAN_SHARED Voxel buffer[Block::voxel_count];

  // set voxel indices
  const int x = threadIdx.x;
  const int y = threadIdx.y;
  const int z = threadIdx.z;

  // compute voxel memory step sizes
  const int voxel_step_y = Block::resolution;
  const int voxel_step_z = Block::resolution * Block::resolution;

  // load voxel data into shared memory
  const int entry_index = visible_blocks[block_index];
  const HashEntry entry = hash_entries[entry_index];
  const size_t block_offset = entry.data * Block::voxel_count;
  const size_t voxel_offset = z * voxel_step_z + y * voxel_step_y + x;
  const size_t voxel_index = block_offset + voxel_offset;
  buffer[voxel_offset] = voxels[voxel_index];

  // const Voxel voxel = voxels[voxel_index];
  // if (fabsf(voxel.distance) < 0.5)
  // {
  //   const Block block = entry.block;

  //   const Vector3f point = 0.008 * ((Block::resolution - 1) *
  //       Vector3f(block.GetOrigin()) + Vector3f(x, y, z));

  //   printf("%f %f %f %f\n", point[0], point[1], point[2], voxel.distance);
  // }

  // initialize cube state
  CubeState state;
  state.coords[0] = x;
  state.coords[1] = y;
  state.coords[2] = z;
  state.state = 0;

  // wait for all block threads to finish
  __syncthreads();

  bool unknown = false;

  // process each cube corner
  for (int i = 0; i <= 1; ++i)
  {
    const int zoff = (z + i) * voxel_step_z;

    for (int j = 0; j <= 1; ++j)
    {
      const int yoff = zoff + (y + j) * voxel_step_y;

      for (int k = 0; k <= 1; ++k)
      {
        // compute voxel index
        const int xoff = yoff + (x + k);

        // shift bitmask for next corner
        // state.state <<= 1;
        state.state >>= 1;

        // check if valid voxel index
        if (xoff < Block::voxel_count)
        {
          // update state bitmask for current corner
          const Voxel voxel = buffer[xoff];
          // state.state |= voxel.distance > 0;
          state.state |= (voxel.distance > 0) << 7;

          if (voxel.weight == 0)
          {
            unknown = true;
          }
        }
      }
    }
  }

  if (unknown) state.state = 0;
  // if (state.state != 0 && state.state != 255) state.state = 1; // TODO: remove!

  // compute output buffer index
  const int value = !state.IsEmpty();
  const int offset = PrefixSum<BLOCK_SIZE>(value, voxel_offset, *state_count);

  // write state to output buffer
  if (offset >= 0) states[offset] = state;
}

template <int BLOCK_SIZE>
VULCAN_GLOBAL
void ExtractVertexEdgesKernel(const CubeState* states, VertexEdge* edges,
    int* edge_count, int cube_count)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  int& total = *edge_count;
  VertexEdge buffer[3];
  int value = 0;

  if (index < cube_count)
  {
    const CubeState state = states[index];
    value = edge_counts[state.state];

    if (value > 0)
    {
      const Vector4c* vec_edge_indices;
      vec_edge_indices = reinterpret_cast<const Vector4c*>(edge_indices);
      Vector4c indices = vec_edge_indices[state.state];

      if (state.coords[0] + 1 == Block::resolution &&
          state.coords[1] + 1 == Block::resolution &&
          state.coords[2] + 1 == Block::resolution)
      {
        value = 0;
      }

      if (state.coords[0] == Block::resolution - 1)
      {
        for (int ii = 0; ii < value; ++ii)
        {
          if (indices[ii] == 0)
          {
            for (int jj = ii; jj < value; ++jj)
            {
              indices[jj] = indices[jj + 1];
            }

            --value;
            break;
          }
        }
      }

      if (state.coords[1] == Block::resolution - 1)
      {
        for (int ii = 0; ii < value; ++ii)
        {
          if (indices[ii] == 3)
          {
            for (int jj = ii; jj < value; ++jj)
            {
              indices[jj] = indices[jj + 1];
            }

            --value;
            break;
          }
        }
      }

      if (state.coords[2] == Block::resolution - 1)
      {
        for (int ii = 0; ii < value; ++ii)
        {
          if (indices[ii] == 8)
          {
            for (int jj = ii; jj < value; ++jj)
            {
              indices[jj] = indices[jj + 1];
            }

            --value;
            break;
          }
        }
      }

      for (int i = 0; i < value; ++i)
      {
        buffer[i].coords = state.coords;
        buffer[i].edge = indices[i];
      }
    }
  }

  const int offset = PrefixSum<BLOCK_SIZE>(value, threadIdx.x, total);

  for (int i = 0; i < value; ++i)
  {
    edges[offset + i] = buffer[i];
  }
}

VULCAN_GLOBAL
void ExtractVertexPointsKernel(int block_index, const int* visible_blocks,
    const HashEntry* hash_entries, const Voxel* voxels, const VertexEdge* edges,
    Vector3f* points, float voxel_length, int edge_count)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < edge_count)
  {
    // const VertexEdge edge = edges[index];
    // const int entry_index = visible_blocks[block_index];
    // const HashEntry entry = hash_entries[entry_index];

    // const Vector3f origin = voxel_length *
    //     ((Block::resolution - 1) * Vector3f(entry.block.GetOrigin()) +
    //      Vector3f(edge.coords));

    // points[index] = origin;

    const VertexEdge edge = edges[index];
    const int entry_index = visible_blocks[block_index];
    const HashEntry entry = hash_entries[entry_index];

    const Vector3f origin = voxel_length *
        ((Block::resolution - 1) * Vector3f(entry.block.GetOrigin()) +
        Vector3f(edge.coords));

    const Vector4c* coords = reinterpret_cast<const Vector4c*>(edge_coords);
    const Vector3s ac = Vector3s(Vector3c(coords[8 * edge.edge + 0]));
    const Vector3s bc = Vector3s(Vector3c(coords[8 * edge.edge + 4]));

    // const Vector3i ai = Vector3i(edge.coords) + Vector3i(ac);
    // const Vector3i bi = Vector3i(edge.coords) + Vector3i(bc);

    // const int offset = entry.data * Block::voxel_count;
    // const int r = Block::resolution;
    // const int r2 = Block::resolution * Block::resolution;
    // const int voxel_index_a = offset + ai[2] * r2 + ai[1] * r + ai[0];
    // const int voxel_index_b = offset + bi[2] * r2 + bi[1] * r + bi[0];

    const Vector3f ao = voxel_length * Vector3f(ac);
    const Vector3f bo = voxel_length * Vector3f(bc);

    const Vector3f ap = origin + ao;
    const Vector3f bp = origin + bo;

    // const Voxel a = voxels[voxel_index_a];
    // const Voxel b = voxels[voxel_index_b];

    // const float ad = fabsf(a.distance);
    // const float bd = fabsf(b.distance);

    // const float alpha = (ad + bd < 1E-8f) ? 0.5f : ad / (ad + bd);
    // const float beta = 1.0f - alpha;

    // if (isnan(alpha))
    // {
    //   printf("ab: %f, bd: %f\n", ad, bd);
    // }

    // if ((a.distance > 0 && b.distance > 0) ||
    //     (a.distance < 0 && b.distance < 0))
    // {
    //   printf("ab: %f, bd: %f\n", ad, bd);
    // }

    // points[index] = alpha * ap + beta * bp;

    points[index] = 0.5f * ap + 0.5f * bp;
  }
}

VULCAN_GLOBAL
void ExtractVertexIndicesKernel()
{
  // ALLOCATAE: index [voxel_count][3] (for [1, 4, 9] edges)

  // for each vertex edge

    // compute output buffer index given coords

    // store kernel thread id in output buffer
}

VULCAN_GLOBAL
void ExtractFacesKernel()
{
  // for each non-empty cube (including padding)

    // if valid cube (not padding)

      // get face count

      // for each face

        // get face edges

        // for each edge

          // get vertex index from map

        // output face indices
}

BlockExtractor::BlockExtractor(std::shared_ptr<const Volume> volume) :
  volume_(volume),
  block_index_(0)
{
  Initialize();
}

std::shared_ptr<const Volume> BlockExtractor::GetVolume() const
{
  return volume_;
}

int BlockExtractor::GetBlockIndex() const
{
  return block_index_;
}

void BlockExtractor::SetBlockIndex(int index)
{
  VULCAN_DEBUG(index >= 0);
  block_index_ = index;
}

void BlockExtractor::Extract(DeviceMesh& mesh)
{
  ExtractCubeStates();

  if (!states_.IsEmpty())
  {
    ExtractVertexEdges();

    if (!edges_.IsEmpty())
    {
      ExtractVertexPoints();
      ExtractVertexIndices();
      ExtractFaces();
      CopyPoints(mesh);
      CopyFaces(mesh);
    }
  }
}

void BlockExtractor::Extract(Mesh& mesh)
{
  ExtractCubeStates();

  if (!states_.IsEmpty())
  {
    ExtractVertexEdges();

    if (!edges_.IsEmpty())
    {
      ExtractVertexPoints();
      ExtractVertexIndices();
      ExtractFaces();
      CopyPoints(mesh);
      CopyFaces(mesh);
    }
  }
}

void BlockExtractor::ExtractCubeStates()
{
  const size_t blocks = 1;
  const size_t resolution = Block::resolution;
  const dim3 threads(resolution, resolution, resolution);

  // TODO: replace with "all allocated" blocks
  const Buffer<int>& visible_block_buffer = volume_->GetVisibleBlocks();
  const Buffer<HashEntry>& hash_entry_buffer = volume_->GetHashEntries();
  const Buffer<Voxel>& voxel_buffer = volume_->GetVoxels();
  const int* visible_blocks = visible_block_buffer.GetData();
  const HashEntry* hash_entries = hash_entry_buffer.GetData();
  const Voxel* voxels = voxel_buffer.GetData();
  CubeState* states = states_.GetData();
  int* state_count = size_.GetData();

  ResetSizePointer();

  CUDA_LAUNCH(ExtractCubeStateKernel<Block::voxel_count>, blocks, threads, 0, 0,
      visible_blocks, block_index_, hash_entries, voxels, states, state_count);

  states_.Resize(GetSizePointer());
}

void BlockExtractor::ExtractVertexEdges()
{
  // TODO: handle states_.size > 1024

  // const size_t blocks = 1;
  // const size_t threads = states_.GetSize();

  const size_t threads = 512;
  const size_t total = states_.GetSize();
  const size_t blocks = GetKernelBlocks(total, threads);

  const CubeState* states = states_.GetData();
  VertexEdge* edges = edges_.GetData();
  int* edge_count = size_.GetData();

  ResetSizePointer();

  CUDA_LAUNCH(ExtractVertexEdgesKernel<512>, blocks, threads,
      0, 0, states, edges, edge_count, total);

  edges_.Resize(GetSizePointer());
}

void BlockExtractor::ExtractVertexPoints()
{
  points_.Resize(edges_.GetSize());

  const size_t threads = 512;
  const size_t total = points_.GetSize();
  const size_t blocks = GetKernelBlocks(total, threads);

  const float voxel_length = volume_->GetVoxelLength();
  const int* visible_blocks = volume_->GetVisibleBlocks().GetData();
  const HashEntry* hash_entries = volume_->GetHashEntries().GetData();
  const Voxel* voxels = volume_->GetVoxels().GetData();
  const VertexEdge* edges = edges_.GetData();

  Vector3f* points = points_.GetData();

  CUDA_LAUNCH(ExtractVertexPointsKernel, blocks, threads, 0, 0, block_index_,
      visible_blocks, hash_entries, voxels, edges, points, voxel_length, total);
}

void BlockExtractor::ExtractVertexIndices()
{
  const size_t blocks = 1;
  const size_t threads = 1;
  CUDA_LAUNCH(ExtractVertexIndicesKernel, blocks, threads, 0, 0);
}

void BlockExtractor::ExtractFaces()
{
  const size_t blocks = 1;
  const size_t threads = 1;
  CUDA_LAUNCH(ExtractFacesKernel, blocks, threads, 0, 0);
}

void BlockExtractor::CopyPoints(DeviceMesh& mesh)
{
  const size_t offset = mesh.points.GetSize();
  const size_t new_size = offset + points_.GetSize();
  VULCAN_DEBUG(new_size <= mesh.points.GetCapacity());
  mesh.points.Resize(new_size);
  points_.CopyToDevice(mesh.points.GetData() + offset);
}

void BlockExtractor::CopyFaces(DeviceMesh& mesh)
{
  const size_t offset = mesh.faces.GetSize();
  const size_t new_size = offset + faces_.GetSize();
  VULCAN_DEBUG(new_size <= mesh.faces.GetCapacity());
  mesh.faces.Resize(new_size);
  faces_.CopyToDevice(mesh.faces.GetData() + offset);
}

void BlockExtractor::CopyPoints(Mesh& mesh)
{
  const size_t offset = mesh.points.size();
  const size_t new_size = offset + points_.GetSize();
  VULCAN_DEBUG(new_size <= mesh.points.capacity());
  mesh.points.resize(new_size);
  points_.CopyToHost(mesh.points.data() + offset);
}

void BlockExtractor::CopyFaces(Mesh& mesh)
{
  const size_t offset = mesh.faces.size();
  const size_t new_size = offset + faces_.GetSize();
  VULCAN_DEBUG(new_size <= mesh.faces.capacity());
  mesh.faces.resize(new_size);
  faces_.CopyToHost(mesh.faces.data() + offset);
}

void BlockExtractor::ResetSizePointer()
{
  const int value = 0;
  const cudaMemcpyKind kind = cudaMemcpyHostToDevice;
  CUDA_DEBUG(cudaMemcpy(size_.GetData(), &value, sizeof(int), kind));
}

int BlockExtractor::GetSizePointer()
{
  int value;
  const cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
  CUDA_DEBUG(cudaMemcpy(&value, size_.GetData(), sizeof(int), kind));
  return value;
}

void BlockExtractor::Initialize()
{
  CreateVoxelStateBuffer();
  CreateVertexEdgeBuffer();
  CreateVertexPointBuffer();
  CreateVertexIndexBuffer();
  CreateFaceBuffer();
  CreateSizePointer();
}

void BlockExtractor::CreateVoxelStateBuffer()
{
  states_.Reserve(Block::voxel_count);
}

void BlockExtractor::CreateVertexEdgeBuffer()
{
  const int edges_per_corner = 3;
  const int corner_count = Block::voxel_count;
  edges_.Reserve(edges_per_corner * corner_count);
}

void BlockExtractor::CreateVertexPointBuffer()
{
  points_.Reserve(edges_.GetCapacity());
}

void BlockExtractor::CreateVertexIndexBuffer()
{
  indices_.Reserve(edges_.GetCapacity());
}

void BlockExtractor::CreateFaceBuffer()
{
  const int max_faces_per_cube = 4;
  faces_.Reserve(max_faces_per_cube * states_.GetCapacity());
}

void BlockExtractor::CreateSizePointer()
{
  size_.Resize(1);
}

////////////////////////////////////////////////////////////////////////////////

Extractor::Extractor(std::shared_ptr<const Volume> volume) :
  volume_(volume)
{
}

std::shared_ptr<const Volume> Extractor::GetVolume() const
{
  return volume_;
}

void Extractor::Extract(DeviceMesh& mesh) const
{
  // TODO: replace with "all allocated" blocks
  const Buffer<int>& blocks = volume_->GetVisibleBlocks();
  const int count = blocks.GetSize();
  BlockExtractor extractor(volume_);
  ResizeMesh(mesh);

  for (int i = 0; i < count; ++i)
  {
    extractor.SetBlockIndex(i);
    extractor.Extract(mesh);
  }
}

void Extractor::Extract(Mesh& mesh) const
{
  // TODO: replace with "all allocated" blocks
  const Buffer<int>& blocks = volume_->GetVisibleBlocks();
  const int count = blocks.GetSize();
  BlockExtractor extractor(volume_);
  ResizeMesh(mesh);

  for (int i = 0; i < count; ++i)
  {
    extractor.SetBlockIndex(i);
    extractor.Extract(mesh);
  }
}

void Extractor::ResizeMesh(DeviceMesh& mesh) const
{
  // TODO: replace with "all allocated" blocks
  const Buffer<int>& blocks = volume_->GetVisibleBlocks();
  const int block_count = blocks.GetSize();

  const int edges_per_corner = 3;
  const int max_corner_count = Block::voxel_count;
  const int max_point_count = edges_per_corner * max_corner_count;
  mesh.points.Reserve(block_count * max_point_count);
  mesh.points.Resize(0);

  const int max_faces_per_cube = 5;
  const int max_cube_count = std::pow(Block::resolution - 1, 3);
  const int max_face_count = max_faces_per_cube * max_cube_count;
  mesh.faces.Reserve(block_count * max_face_count);
  mesh.faces.Resize(0);
}

void Extractor::ResizeMesh(Mesh& mesh) const
{
  // TODO: replace with "all allocated" blocks
  const Buffer<int>& blocks = volume_->GetVisibleBlocks();
  const int block_count = blocks.GetSize();

  const int edges_per_corner = 3;
  const int max_corner_count = Block::voxel_count;
  const int max_point_count = edges_per_corner * max_corner_count;
  mesh.points.reserve(block_count * max_point_count);
  mesh.points.resize(0);

  const int max_faces_per_cube = 5;
  const int max_cube_count = std::pow(Block::resolution - 1, 3);
  const int max_face_count = max_faces_per_cube * max_cube_count;
  mesh.faces.reserve(block_count * max_face_count);
  mesh.faces.resize(0);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace vulcan