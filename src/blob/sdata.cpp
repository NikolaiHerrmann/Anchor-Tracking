#include "blob.hpp"

int *Blob::s_points = new int[WIDTH * HEIGHT];
size_t Blob::s_maxID = 0;