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
    enum Param
    {
        BOX = 13,
        WIDTH = 640,
        HEIGHT = 480,
    };

    int d_minX;
    int d_minY;
    int d_maxX;
    int d_maxY;
    int d_id;
    size_t d_count;

    static int *s_points;
    static size_t s_maxID;

    public:
        Blob(int x, int y);
        void add(int x, int y);

        bool inBound(int x, int y);
        void color(cv::Mat &image);

        cv::Rect rectangle();
        void draw(cv::Mat &image);

        static void clear();
        static void destroy();
        size_t count();

        static void print();
    
    private:
        void floodFill();
};

inline void Blob::clear()
{
    std::fill(s_points, s_points + (WIDTH * HEIGHT), -1);
    s_maxID = 0;
}

inline size_t Blob::count()
{
    return d_count;
}

inline cv::Rect Blob::rectangle()
{
    return cv::Rect{d_minX, d_minY, d_maxX - d_minX, d_maxY - d_minY};
}

inline void Blob::destroy()
{
    delete[] s_points;
}

#endif