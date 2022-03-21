#include "tracker.hpp"
#include <opencv2/opencv.hpp>

void Tracker::initFrame()
{
    d_count = 0;
    d_avgX = 0;
    d_avgY = 0;
}

void Tracker::perPixel(size_t x, size_t y, cv::Vec3b const &rgb)
{
    cv::Vec3b tc{255, 0, 0};
    size_t dis = colorDistance(tc, rgb);

    if (dis < d_colorThreshold * d_colorThreshold)
    {
        d_avgX += x;
        d_avgY += y;
        d_count++;
    }
}

void Tracker::perFrame()
{
    if (d_count > 0)
    {
        d_avgX /= d_count;
        d_avgY /= d_count;
        cv::Rect rec(d_avgX-25, d_avgY-25, 50, 50);
        cv::rectangle(d_image, rec, cv::Scalar(0, 0, 255));
    }
}

size_t Tracker::colorDistance(cv::Vec3b const &c1, cv::Vec3b const &c2)
{
    return (c2[0] - c1[0]) * (c2[0] - c1[0]) + (c2[1] - c1[1]) * (c2[1] - c1[1]) + (c2[2] - c1[2]) * (c2[2] - c1[2]);
}