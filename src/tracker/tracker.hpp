#ifndef INCLUDED_TRACKER_H
#define INCLUDED_TRACKER_H

#include "../video/video.hpp"

class Tracker: public Video
{
    int d_count;
    size_t d_avgX;
    size_t d_avgY;
    size_t d_colorThreshold = 180;

    private:
        void perPixel(size_t x, size_t y, cv::Vec3b const &rgb) override;
        void initFrame() override;
        void perFrame() override;

        size_t colorDistance(cv::Vec3b const &c1, cv::Vec3b const &c2);
};

#endif