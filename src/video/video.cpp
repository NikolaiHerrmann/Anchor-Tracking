#include "video.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

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

        d_tracker.scan(d_image);

        cv::imshow("", d_image);

        if (cv::waitKey(10) == ESC)
            break;
    }
}