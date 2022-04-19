#ifndef INCLUDED_TRACKER_H
#define INCLUDED_TRACKER_H

#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include "../blob/blob.hpp"
#include <vector>

class Tracker
{
    cv::Mat d_image;
    cv::Mat d_prev;
    std::vector<Blob> d_blobs;

    public:
        void scan(cv::Mat &image);

    private:
        void binarize(cv::Mat const &image);
        void findBlob(uint8_t intensity, int x, int y);
};

#endif