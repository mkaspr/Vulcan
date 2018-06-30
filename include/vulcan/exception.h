#pragma once

#include <exception>
#include <string>

#ifdef __CUDA_ARCH__
#define VULCAN_THROW(text)                                                     \
  printf("[ %u %u %u ][ %u %u %u ] %s(%u): %s\n",                              \
  blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,   \
  __FILE__, __LINE__, text)
#else
#define VULCAN_THROW(text)                                                     \
  throw ::vulcan::Exception(__LINE__, __FILE__, text)
#endif

#define VULCAN_ASSERT_MSG(cond, text)                                          \
  if (!(cond)) VULCAN_THROW(text)

#define VULCAN_ASSERT(cond)                                                    \
  if (!(cond)) VULCAN_THROW("assertion failed: " #cond)

#ifdef NDEBUG
#define VULCAN_DEBUG_MSG(cond, text)
#define VULCAN_DEBUG(cond)
#else
#define VULCAN_DEBUG_MSG VULCAN_ASSERT_MSG
#define VULCAN_DEBUG VULCAN_ASSERT
#endif

namespace vulcan
{

class Exception : public std::exception
{
  public:

    Exception(int line, const std::string& file, const std::string& text) :
      line_(line),
      file_(file),
      text_(text)
    {
      Initialize();
    }

    inline int line() const
    {
      return line_;
    }

    inline const std::string& file() const
    {
      return file_;
    }

    inline const std::string& text() const
    {
      return text_;
    }

    inline const char* what() const throw() override
    {
      return what_.c_str();
    }

  private:

    void Initialize()
    {
      what_ = file_ + "(" + std::to_string(line_) + "): " + text_;
    }

  protected:

    int line_;

    std::string file_;

    std::string text_;

    std::string what_;
};

} // namespace vulcan