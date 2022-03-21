#include "tracker.hpp"
#include <opencv2/opencv.hpp>

void Tracker::initFrame()
{
    d_count = 0;
    d_avgX = 0;
    d_avgY = 0;
}

void Tracker::perPixel(Pixel const &rgb, int x, int y)
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

void Tracker::perFrame(cv::Mat const &image)
{
    if (d_count > 0)
    {
        d_avgX /= d_count;
        d_avgY /= d_count;
        cv::Rect rec(d_avgX-25, d_avgY-25, 50, 50);
        cv::rectangle(image, rec, cv::Scalar(0, 0, 255));
    }
}

size_t Tracker::colorDistance(Pixel const &c1, Pixel const &c2)
{
    return (c2.x - c1.x) * (c2.x - c1.x) + (c2.y - c1.y) * (c2.y - c1.y) + (c2.z - c1.z) * (c2.z - c1.z);
}