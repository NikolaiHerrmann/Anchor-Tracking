#include "video.hpp"
#include <opencv2/highgui.hpp>

Video::Video(size_t cameraIdx, size_t width, size_t height)
:
    d_width(width),
    d_height(height),
    d_capture(cameraIdx)
{
    d_capture.set(cv::CAP_PROP_FRAME_WIDTH, d_width);
    d_capture.set(cv::CAP_PROP_FRAME_HEIGHT, d_height);
}

void Video::run()
{
    while (true)
    {
        bool readStatus = d_capture.read(d_image);

        if (!readStatus)
            throw "Frame read failed!\n";

        d_tracker.initFrame();

        d_image.forEach<Tracker::Pixel>(
            [&](Tracker::Pixel &pixel, int const *pos)
            {
                d_tracker.perPixel(pixel, pos[1], pos[0]);
            }
        );

        d_tracker.perFrame(d_image);

        cv::imshow("", d_image);

        if (cv::waitKey(10) == ESC)
            break;
    }
}