#include <nanobind/nanobind.h>

// Include the header files for the functions we want to export here.

int sub(int a, int b) { return a - b; }
int add(int a, int b) { return a + b; }

NB_MODULE(_bsparse, m)
{
    m.def("sub", &sub);
    m.def("add", &add);
}