#include "complex.h"
#include "convolution.h"
#include "array"

// Compile with:
// g++ -I .. -fopenmp exampleconv.cc ../convolution.cc ../fftw++.cc -lfftw3 -lfftw3_omp

using namespace std;
using namespace utils;
using namespace Array;
using namespace fftwpp;

inline void init(complex *f, complex *g, unsigned int m)
{
  for(unsigned int k=0; k < m; k++) {
    f[k]=complex(k,k+1);
    g[k]=complex(k,2*k+1);
  }
}

int main(int argc, char* argv[])
{
  fftw::maxthreads=get_max_threads();

  // size of problem
  unsigned int m=8;

#ifndef __SSE2__
  fftw::effort |= FFTW_NO_SIMD;
#endif

  // 1d centered Hermitian-symmetric complex convolution
  cout << "1D centered Hermitian-symmetric convolution:" << endl;

  // allocate arrays:
  complex *f=complexAlign(m);
  complex *g=complexAlign(m);

  init(f,g,m);
  cout << "\ninput:\nf\tg" << endl;
  for(unsigned int i=0; i < m; i++)
    cout << f[i] << "\t" << g[i] << endl;

  ImplicitHConvolution C(m);
  C.convolve(f,g);

  cout << "\noutput:" << endl;
  for(unsigned int i=0; i < m; i++) cout << f[i] << endl;

  deleteAlign(g);
  deleteAlign(f);

  return 0;
}
