#include "tracker.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

void Tracker::binarize(cv::Mat &image)
{
    cv::cvtColor(image, d_image, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(d_image, d_image, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 9, 11);

    cv::dilate(d_image, d_image, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

    cv::erode( d_image, d_image, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);
    cv::bitwise_not(d_image, d_image);
    
    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(d_image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    for (size_t i = 0; i < contours.size(); ++i)
        cv::drawContours(d_image, contours, i, cv::Scalar(0), 2);

    if (d_prev.empty())
        d_prev = d_image.clone();
        
    cv::Mat cp = d_image.clone();
    
    cv::bitwise_and(d_image, d_prev, d_image);
    d_prev = cp.clone();
}

void Tracker::findBlob(uint8_t intensity, int x, int y)
{
    if (intensity != 0)
        return;

    for (size_t i = 0; i < d_blobs.size(); ++i)
    {
        if (d_blobs[i]->inBound(x, y))
        {
            d_blobs[i]->add(x, y);
            return;
        }
    }

    d_blobs.push_back(std::shared_ptr<Blob>(new Blob{x, y}));
}

void Tracker::scan(cv::Mat &image)
{
    d_blobs.clear();
    Blob::clear();

    binarize(image);

    for (int i = BORDER_OFFSET; i < d_image.rows - BORDER_OFFSET; ++i)
    {
        uint8_t* pixel = d_image.ptr<uint8_t>(i);
        for (int j = BORDER_OFFSET; j < d_image.cols - BORDER_OFFSET; ++j)
            findBlob(pixel[j], j, i);
    }

    image = d_image;

    std::vector<std::shared_ptr<Blob>> filter;

    for (size_t i = 0; i < d_blobs.size(); ++i) 
    {
        if (d_blobs[i]->count() > 100)
        {
            d_blobs[i]->draw(image);
            filter.push_back(d_blobs[i]);
        }
    }

    // Blob::print();
    // exit(1);
    // std::cout << filter.size() << "\n";
}