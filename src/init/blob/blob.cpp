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
    d_id(s_maxID++),
    d_count(0)
{
    add(x, y);
}

void Blob::add(int x, int y)
{
    d_minX = min(d_minX, x);
    d_minY = min(d_minY, y);
    d_maxX = max(d_maxX, x);
    d_maxY = max(d_maxY, y);
    ++d_count;

    int offset = BOX / 2;
    int yMinLimit = max(y - offset, 0);
    int yMaxLimit = min(y + offset, HEIGHT);

    int xMinLimit = max(x - offset, 0);
    int xMaxLimit = min(x + offset, WIDTH);

    int xDist = xMaxLimit - xMinLimit;

    int *points = s_points + (xMinLimit + (yMinLimit * WIDTH));

    for (int i = yMinLimit; i != yMaxLimit; ++i, points += WIDTH)
        std::fill(points, points + xDist, d_id);
}

void Blob::print()
{
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            std::cout << *(s_points + (j + (i * WIDTH))) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n\n\n";
}

bool Blob::inBound(int x, int y)
{
    return s_points[x + (y * WIDTH)] == d_id;
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

    // cv::putText(image, sstream.str(), 
    //             cv::Point(d_minX, d_minY), cv::FONT_HERSHEY_COMPLEX_SMALL, 
    //             0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
}