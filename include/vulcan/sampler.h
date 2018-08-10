#pragma once

namespace vulcan
{

class ColorImage;
class Image;

class Sampler
{
  public:

    enum Filter
    {
      FILTER_NEAREST,
      FILTER_LINEAR,
    };

  public:

    Sampler();

    void GetSubimage(const Image& image, Image& subimage,
        Filter filter = FILTER_NEAREST) const;

    void GetSubimage(const ColorImage& image, ColorImage& subimage,
        Filter filter = FILTER_NEAREST) const;

    void GetGradient(const Image& image, Image& gradient_x,
        Image& gradient_y) const;

    void GetGradient(const ColorImage& image, ColorImage& gradient_x,
        ColorImage& gradient_y) const;
};

} // namespace vulcan