#ifndef INCLUDED_VIDEO_H
#define INCLUDED_VIDEO_H

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/types.hpp>
#include "../tracker/tracker.hpp"

class Video
{
    size_t d_width;
    size_t d_height;
    cv::VideoCapture d_capture;
    cv::Mat d_image;
    Tracker d_tracker;

    enum Action
    {
        ESC = 27
    };

    public:
        explicit Video(size_t cameraIdx = 2, size_t width = 640, size_t height = 480);
        void run();
};

#endif