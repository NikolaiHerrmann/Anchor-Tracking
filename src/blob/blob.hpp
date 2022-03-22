#ifndef INCLUDED_BLOB_H
#define INCLUDED_BLOB_H

#include <opencv2/opencv.hpp>
#include <cmath>

class Blob
{
    enum Threshold
    {
        DISTANCE = 70
    };

    const int d_threshSq = DISTANCE * DISTANCE;

    int d_minX;
    int d_minY;
    int d_maxX;
    int d_maxY;

    public:
        Blob(int x, int y);

        void add(int x, int y);

        inline bool inBound(int x, int y)
        {
            return distanceSq((d_maxX + d_minX) >> 1, (d_maxY + d_minY) >> 1, x, y) < d_threshSq;
        }

        constexpr const int distanceSq(int x1, int y1, int x2, int y2)
        {
            return std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2);
        }

        inline cv::Rect rectangle()
        {
            return cv::Rect{d_minX, d_minY, d_maxX - d_minX, d_maxY - d_minY};
        }
};

#endif