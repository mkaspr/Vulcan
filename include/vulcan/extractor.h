#pragma once

#include <memory>
#include <vulcan/buffer.h>
#include <vulcan/matrix.h>

namespace vulcan
{

class Mesh;
class Volume;

struct VoxelState
{
  unsigned char index;

  unsigned char state;
};

struct VertexEdge
{
  unsigned char index;

  unsigned char edge;
};

class BlockExtractor
{
  public:

    BlockExtractor(std::shared_ptr<const Volume> volume);

    std::shared_ptr<const Volume> GetVolume() const;

    int GetBlockIndex() const;

    void SetBlockIndex(int index);

    void Extract(Mesh& mesh);

  protected:

    void ExtractVoxelState();

    void ExtractVertexEdges();

    void ExtractVertexPoints();

    void ExtractVertexIndices();

    void ExtractFaces();

    void CopyPoints(Mesh& mesh);

    void CopyFaces(Mesh& mesh);

  private:

    void Initialize();

    void CreateVoxelStateBuffer();

    void CreateVertexEdgesBuffer();

    void CreateVertexPointsBuffer();

    void CreateVertexIndicesBuffer();

    void CreateFacesBuffer();

  protected:

    std::shared_ptr<const Volume> volume_;

    Buffer<VoxelState> state_;

    Buffer<VertexEdge> edges_;

    Buffer<Vector3f> points_;

    Buffer<int> indices_;

    Buffer<Vector3i> faces_;

    int block_index_;
};

class Extractor
{
  public:

    Extractor(std::shared_ptr<const Volume> volume);

    std::shared_ptr<const Volume> GetVolume() const;

    void Extract(Mesh& mesh) const;

  protected:

    void ResizeMesh(Mesh& mesh) const;

  protected:

    std::shared_ptr<const Volume> volume_;
};

} // namespace vulcan