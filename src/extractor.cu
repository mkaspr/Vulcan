#include <vulcan/extractor.h>
#include <vulcan/block.h>
#include <vulcan/exception.h>
#include <vulcan/volume.h>

namespace vulcan
{

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

void BlockExtractor::Extract(Mesh& mesh)
{
  ExtractVoxelState();
  ExtractVertexEdges();
  ExtractVertexPoints();
  ExtractVertexIndices();
  ExtractFaces();
  CopyPoints(mesh);
  CopyFaces(mesh);
}

void BlockExtractor::ExtractVoxelState()
{
}

void BlockExtractor::ExtractVertexEdges()
{
}

void BlockExtractor::ExtractVertexPoints()
{
}

void BlockExtractor::ExtractVertexIndices()
{
}

void BlockExtractor::ExtractFaces()
{
}

void BlockExtractor::CopyPoints(Mesh& mesh)
{
}

void BlockExtractor::CopyFaces(Mesh& mesh)
{
}

void BlockExtractor::Initialize()
{
  CreateVoxelStateBuffer();
  CreateVertexEdgesBuffer();
  CreateVertexPointsBuffer();
  CreateVertexIndicesBuffer();
  CreateFacesBuffer();
}

void BlockExtractor::CreateVoxelStateBuffer()
{
  state_.Reserve(std::pow(Block::resolution - 1, 3));
}

void BlockExtractor::CreateVertexEdgesBuffer()
{
  const int edge_count = 12;
  edges_.Reserve(edge_count * state_.GetCapacity());
}

void BlockExtractor::CreateVertexPointsBuffer()
{
  points_.Reserve(edges_.GetCapacity());
}

void BlockExtractor::CreateVertexIndicesBuffer()
{
}

void BlockExtractor::CreateFacesBuffer()
{
}

Extractor::Extractor(std::shared_ptr<const Volume> volume) :
  volume_(volume)
{
}

std::shared_ptr<const Volume> Extractor::GetVolume() const
{
  return volume_;
}

void Extractor::Extract(Mesh& mesh) const
{
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

void Extractor::ResizeMesh(Mesh& mesh) const
{
  VULCAN_THROW("not implemented");
}

} // namespace vulcan