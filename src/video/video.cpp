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

        initFrame();

        // d_image.forEach(
        //     [](cv::vec3b)
        // )

        for (size_t row = 0; row < d_image.rows; ++row)
        {
            cv::Vec3b *pixel = d_image.ptr<cv::Vec3b>(row);

            for (size_t col = 0; col < d_image.cols; ++col)
                perPixel(col, row, pixel[col]);
        }

        perFrame();

        cv::imshow("", d_image);

        if (cv::waitKey(10) == ESC)
            break;
    }
}