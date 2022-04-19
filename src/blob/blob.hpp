#ifndef INCLUDED_BLOB_H
#define INCLUDED_BLOB_H

#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>

#define max(a, b) a > b ? a : b
#define min(a, b) a < b ? a : b

class Blob
{
    enum Threshold
    {
        DISTANCE = 50,
        BOX = 9,
        WIDTH = 640,
        HEIGHT = 480,
    };

    int const d_threshSq = DISTANCE * DISTANCE;

    int d_minX;
    int d_minY;
    int d_maxX;
    int d_maxY;
    int d_id;

    static int *s_points;
    static size_t s_maxID;

    public:
        Blob(int x, int y);
        void add(int x, int y);
        int cx();
        int cy();
        bool inBound(int x, int y);
        int distanceSq(int x1, int y1, int x2, int y2);

        int size();
        cv::Point position();
        void color(cv::Mat &image);

        cv::Rect rectangle();
        void draw(cv::Mat &image);

        static void clear();
};

inline void Blob::clear()
{
    std::fill(s_points, s_points + (WIDTH * HEIGHT), -1);
    s_maxID = 0;
}

inline int Blob::cx()
{
    return (d_maxX + d_minX) >> 1;
}

inline int Blob::cy()
{
    return (d_maxY + d_minY) >> 1;
}



inline int Blob::distanceSq(int x1, int y1, int x2, int y2)
{
    return std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2);
}

inline int Blob::size()
{
    return (d_maxX - d_minX) * (d_maxY - d_minY);
}

inline cv::Point Blob::position()
{
    return cv::Point{cx(), cy()};
}

inline cv::Rect Blob::rectangle()
{
    return cv::Rect{d_minX, d_minY, d_maxX - d_minX, d_maxY - d_minY};
}

#endif