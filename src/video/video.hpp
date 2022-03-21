#ifndef INCLUDED_VIDEO_H
#define INCLUDED_VIDEO_H

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

class Video
{
    //typedef Point3_<uint8_t> Pixel;

    enum Action
    {
        ESC = 27
    };

    protected:
        size_t d_width;
        size_t d_height;
        cv::VideoCapture d_capture;
        cv::Mat d_image;

    public:
        explicit Video(size_t cameraIdx = 2, size_t width = 640, size_t height = 480);
        void run();

    private:
        virtual void perPixel(size_t x, size_t y, cv::Vec3b const &rgb) = 0;
        virtual void initFrame() = 0;
        virtual void perFrame() = 0;
};

#endif