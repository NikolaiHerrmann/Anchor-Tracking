#include "video.hpp"
#include <opencv2/highgui.hpp>

Video::Video(size_t cameraIdx, size_t width, size_t height)
:
    d_capture(cameraIdx)
{
    d_capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    d_capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
}

void Video::run()
{
    while (true)
    {
        if (!d_capture.read(d_image))
            throw "Frame read failed!\n";

        d_tracker.init();

        d_image.forEach<Tracker::Pixel>(
            [&](Tracker::Pixel &pixel, int const *pos)
            {
                d_tracker.perPixel(pixel, pos[1], pos[0]);
            }
        );

        d_tracker.draw(d_image);

        cv::imshow("", d_image);

        if (cv::waitKey(10) == ESC)
            break;
    }
}