#pragma once

#include <memory>
#include <vulcan/buffer.h>
#include <vulcan/device.h>
#include <vulcan/matrix.h>

namespace vulcan
{

class DeviceMesh;
class Mesh;
class Volume;

struct CubeState
{
  VULCAN_HOST_DEVICE
  inline bool IsEmpty() const
  {
    return (state == 0x00) | (state == 0xFF);
  }

  Vector3c coords;

  unsigned char state;
};

struct VertexEdge
{
  VULCAN_HOST_DEVICE
  inline bool HasValidCorner() const
  {
    return false;
  }

  Vector3c coords;

  unsigned char edge;
};

class BlockExtractor
{
  public:

    BlockExtractor(std::shared_ptr<const Volume> volume);

    std::shared_ptr<const Volume> GetVolume() const;

    int GetBlockIndex() const;

    void SetBlockIndex(int index);

    void Extract(DeviceMesh& mesh);

    void Extract(Mesh& mesh);

  protected:

    void ExtractCubeStates();

    void ExtractVertexEdges();

    void ExtractVertexPoints();

    void ExtractVertexIndices();

    void ExtractFaces();

    void CopyPoints(DeviceMesh& mesh);

    void CopyFaces(DeviceMesh& mesh);

    void CopyPoints(Mesh& mesh);

    void CopyFaces(Mesh& mesh);

    void ResetSizePointer();

    int GetSizePointer();

  private:

    void Initialize();

    void CreateVoxelStateBuffer();

    void CreateVertexEdgeBuffer();

    void CreateVertexPointBuffer();

    void CreateVertexIndexBuffer();

    void CreateFaceBuffer();

    void CreateSizePointer();

  protected:

    std::shared_ptr<const Volume> volume_;

    Buffer<CubeState> states_;

    Buffer<VertexEdge> edges_;

    Buffer<Vector3f> points_;

    Buffer<int> indices_;

    Buffer<Vector3i> faces_;

    Buffer<int> size_;

    int block_index_;
};

class Extractor
{
  public:

    Extractor(std::shared_ptr<const Volume> volume);

    std::shared_ptr<const Volume> GetVolume() const;

    void Extract(DeviceMesh& mesh) const;

    void Extract(Mesh& mesh) const;

  protected:

    void ResizeMesh(DeviceMesh& mesh) const;

    void ResizeMesh(Mesh& mesh) const;

  protected:

    std::shared_ptr<const Volume> volume_;
};

} // namespace vulcan