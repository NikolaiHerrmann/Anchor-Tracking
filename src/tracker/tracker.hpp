#ifndef INCLUDED_TRACKER_H
#define INCLUDED_TRACKER_H

#include <opencv2/core/types.hpp>

class Tracker
{
    int d_count;
    size_t d_avgX;
    size_t d_avgY;
    size_t d_colorThreshold = 180;

    public:
        typedef cv::Point3_<uint8_t> Pixel;
        
        void perPixel(Pixel const &pixel, int x, int y);
        void init();
        void draw(cv::Mat &image);

    private:
        size_t colorDistance(Pixel const &c1, Pixel const &c2);
};

#endif