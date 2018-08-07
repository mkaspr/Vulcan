#pragma once

#include <memory>
#include <vulcan/buffer.h>
#include <vulcan/matrix.h>

namespace vulcan
{

class Frame;
class Volume;

struct Patch
{
  static const int max_size = 16;

  Vector2s origin;

  Vector2s size;

  Vector2f bounds;
};

class Tracer
{
  public:

    Tracer(std::shared_ptr<const Volume> volume);

    std::shared_ptr<const Volume> GetVolume() const;

    void Trace(Frame& frame);

  protected:

    void ComputePatches(const Frame& frame);

    void ComputeBounds(const Frame& frame);

    void ComputePoints(Frame& frame);

    void ComputeNormals(Frame& frame);

    void ResetBoundsBuffer();

    void ResetBufferSize();

    int GetBufferSize();

  private:

    void Initialize();

    void CreatePatchBuffer();

    void CreateBoundsBuffer();

    void CreateSizeBuffer();

  protected:

    Buffer<Patch> patches_;

    Buffer<Vector2f> bounds_;

    Buffer<int> buffer_size_;

    std::shared_ptr<const Volume> volume_;
};

} // namespace vulcan