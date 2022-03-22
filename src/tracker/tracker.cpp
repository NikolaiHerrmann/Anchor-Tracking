#include "tracker.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

void Tracker::binarize(cv::Mat const &image)
{
    cv::cvtColor(image, d_image, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(d_image, d_image, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 3, 12);
}

void Tracker::findBlob(uint8_t intensity, int x, int y)
{
    if (intensity != 0)
        return;

    for (size_t i = 0; i < d_blobs.size(); ++i)
    {
        if (d_blobs[i].inBound(x, y))
        {
            d_blobs[i].add(x, y);
            return;
        }
    }

    d_blobs.push_back(Blob{x, y});
}

void Tracker::scan(cv::Mat &image)
{
    d_blobs.clear();

    binarize(image);

    for (int i = 0; i < d_image.rows; ++i)
    {
        uint8_t* pixel = d_image.ptr<uint8_t>(i);
        for (int j = 0; j < d_image.cols; ++j)
            findBlob(pixel[j], j, i);
    }

    //image = d_image;

    for (size_t i = 0; i < d_blobs.size(); ++i)
    {
        
        d_blobs[i].draw(image);
    }

}