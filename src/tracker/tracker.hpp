#ifndef INCLUDED_TRACKER_H
#define INCLUDED_TRACKER_H

#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include "../blob/blob.hpp"
#include <vector>

class Tracker
{
    cv::Mat d_image;
    std::vector<Blob> d_blobs;

    public:
        typedef cv::Point3_<uint8_t> PixelRGB;
        
        
        void scan(cv::Mat &image);

    private:
        void binarize(cv::Mat const &image);
        void findBlob(uint8_t intensity, int x, int y);
};

#endif