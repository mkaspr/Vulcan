#pragma once

#include <memory>

namespace vulcan
{

class Frame;
class Volume;

class Tracer
{
  public:

    Tracer(std::shared_ptr<Volume> volume);

    std::shared_ptr<Volume> GetVolume() const;

    void Trace(Frame& frame);

  protected:

    void ComputePatches();

    void ComputeBounds();

    void ResizeFrame(Frame& frame);

    void ComputePoints(Frame& frame);

    void ComputeNormals(Frame& frame);

  private:

    void Initialize();

  protected:

    std::shared_ptr<Volume> volume_;
};

} // namespace vulcan