#include "blob.hpp"
#include <algorithm>

Blob::Blob(int x, int y)
:
    d_minX(x),
    d_minY(y),
    d_maxX(x),
    d_maxY(y)
{
}

void Blob::add(int x, int y)
{
    d_minX = std::min(d_minX, x);
    d_minY = std::min(d_minY, y);
    d_maxX = std::max(d_maxX, x);
    d_maxY = std::max(d_maxY, y);
}