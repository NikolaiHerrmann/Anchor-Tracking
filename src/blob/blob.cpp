#include "blob.hpp"
#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>

Blob::Blob(int x, int y)
:
    d_minX(x),
    d_minY(y),
    d_maxX(x),
    d_maxY(y),
    d_id(rand())
{
}

void Blob::add(int x, int y)
{
    d_minX = std::min(d_minX, x);
    d_minY = std::min(d_minY, y);
    d_maxX = std::max(d_maxX, x);
    d_maxY = std::max(d_maxY, y);
}

void Blob::color(cv::Mat &image)
{

}

void Blob::draw(cv::Mat &image)
{
    cv::Mat crop = image(rectangle());
    cv::Scalar channels = cv::mean(crop);

    std::stringstream sstream{};
    sstream << channels;

    cv::rectangle(image, rectangle(), cv::Scalar(0, 0, 255));
    // "p: (" + std::to_string(cx()) + ", " + std::to_string(cy()) + 
    //              ") s: " + std::to_string(size())

    cv::putText(image, sstream.str(), 
                cv::Point(d_minX, d_minY), cv::FONT_HERSHEY_COMPLEX_SMALL, 
                0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
}