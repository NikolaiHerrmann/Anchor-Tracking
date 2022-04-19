#include "video/video.hpp"
#include <string>

int main(int argc, char **argv)
{
    Video video{argc > 1 ? std::stoul(argv[1]) : 0};
    video.run();
}