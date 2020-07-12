#include "fftw3.h"
#include "math.h"
#include "iostream"
#include <chrono>
using namespace std::chrono;

using namespace std;

void RealToComplexFFT();
void ComplexToComplexFFT();
void BenchmarkTest();

int main()
{
    string choice = "BenchmarkTest";

    if (choice == "ComplexToComplexFFT"){
        ComplexToComplexFFT();
    }
    else if (choice == "RealToComplexFFT"){
        RealToComplexFFT();
    }
    else if (choice == "BenchmarkTest"){
        BenchmarkTest();
    }

    return 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RealToComplexFFT(){

    // INPUT POINTERS

    double Iin[10] = {0.186775, 0.186775, 0.186775, 0.186775, 0.186775, 0, 0.186775, 0.186775, 0.186775, 0.186775};
    const int input_length = sizeof(Iin)/sizeof(Iin[0]);
    double *input = new double[input_length];
    input = Iin;

    double Kin[5] = {0.2, 0.4, 0.6, 0.4, 0.1};
    const int kernel_length = sizeof(Kin)/sizeof(Kin[0]);
    double *ker = new double[kernel_length];
    ker = Kin;


    // FFTW START

    int padding = kernel_length + input_length - 1;
    const double scale = 1./(padding);

    fftw_complex *input_fft;
    fftw_complex *ker_fft;
    double *convolution = new double[padding];
    fftw_complex *convolution_f;
    fftw_plan     p, pinv;


    double *input_padding = new double[padding];
    double *ker_padding = new double[padding];

    // pad input and kernel
    for (int i = 0; i < padding; ++i)
    {
        if (i < kernel_length){
            ker_padding[i] = ker[i];
        }
        else{
            ker_padding[i] = 0;
        }

        if (i < input_length){
            input_padding[i] = input[i];
        }
        else{
            input_padding[i] = 0;
        }
    }

    // compute and compare with expected result
    for (int i = 0; i < padding; i++)
    {
        double expected = 0;

        for (int k = 0; k < padding; k++)
        {
            if((i-k) >=0 && (i-k) < padding){
                expected += input_padding[i-k] * ker_padding[k] ;
            }
        }
        printf("i=%d, Expected: %f \n", i, expected);
    }
    cout << endl;

    ker_fft    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);
    input_fft    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);
    convolution_f    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);

    p = fftw_plan_dft_r2c_1d(padding,
                             ker_padding,
                             ker_fft,
                             FFTW_ESTIMATE);

    fftw_execute(p);

    p = fftw_plan_dft_r2c_1d(padding,
                             input_padding,
                             input_fft,
                             FFTW_ESTIMATE);
    fftw_execute(p);

    // convolution in frequency domain
    for(int i = 0; i < padding; ++i)
    {
        convolution_f[i][0] = (ker_fft[i][0] * input_fft[i][0] - ker_fft[i][1] * input_fft[i][1]) * scale;
        convolution_f[i][1] = (ker_fft[i][0] * input_fft[i][1] + ker_fft[i][1] * input_fft[i][0]) * scale;
        //        cout<< convolution_f[i][0] << " " << convolution_f[i][1] << endl;
    }

    pinv = fftw_plan_dft_c2r_1d(padding,
                                convolution_f,
                                convolution,
                                FFTW_ESTIMATE);
    fftw_execute(pinv);

    // compute and compare with expected result
    for (int i = 0; i < padding; i++)
    {
        printf("i=%d, FFT: r%f:\n", i, convolution[i]);
    }

    fftw_destroy_plan(p);
    fftw_destroy_plan(pinv);

    fftw_free(ker_fft); fftw_free(input_fft); fftw_free(convolution_f);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ComplexToComplexFFT(){

    // Input signal
    double Signal_in[10] = {0.186775, 0.186775, 0.186775, 0.186775, 0.186775, 0, 0.186775, 0.186775, 0.186775, 0.186775};
    const int input_length = sizeof(Signal_in)/sizeof(Signal_in[0]);
    double *input = new double[input_length];
    input = Signal_in;

    // Kernel
    double Ker_in[5] = {0.2, 0.4, 0.6, 0.4, 0.1};
    const int kernel_length = sizeof(Ker_in)/sizeof(Ker_in[0]);
    double *ker = new double[kernel_length];
    ker = Ker_in;

    // Padding length
    int padding = kernel_length + input_length - 1;

    // Normalization factor
    const double scale = 1./(padding);

    double input_padding[padding];
    double ker_padding[padding];

    // pad input and kernel for direct convolution
    for (int i = 0; i < padding; ++i)
    {
        if (i < kernel_length){
            ker_padding[i] = ker[i];
        }
        else{
            ker_padding[i] = 0;
        }

        if (i < input_length){
            input_padding[i] = input[i];
        }
        else{
            input_padding[i] = 0;
        }
    }


    // FFT STARTS HERE

    // Declare complex variables
    fftw_complex *kernel, *inSignal, *convolution;
    fftw_complex *kernel_fft, *inSignal_f, *convolution_f;
    fftw_plan     p, pinv;

    // Allocate memory
    kernel = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);
    inSignal    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);
    convolution = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);
    kernel_fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);
    inSignal_f    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);
    convolution_f    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);


    // Signals to be convolved: Complex kernel and input array filling with padding
    for (int i = 0; i < padding; ++i)
    {
        if (i < kernel_length){
            kernel[i][0] = ker[i];
            kernel[i][1] = 0;
        }
        else{
            kernel[i][0] = 0;
            kernel[i][1] = 0;
        }

        if (i < input_length){
            inSignal[i][0] = input[i];
            inSignal[i][1] = 0;
        }
        else{
            inSignal[i][0] = 0;
            inSignal[i][1] = 0;
        }
    }


    // FFT kernel
    p = fftw_plan_dft_1d(padding,
                         kernel,
                         kernel_fft,
                         FFTW_FORWARD,
                         FFTW_ESTIMATE);

    fftw_execute(p);

    // FFT input
    p = fftw_plan_dft_1d(padding,
                         inSignal,
                         inSignal_f,
                         FFTW_FORWARD,
                         FFTW_ESTIMATE);
    fftw_execute(p);

    // convolution in frequency domain
    for(int i = 0; i < padding; ++i)
    {
        convolution_f[i][0] = (kernel_fft[i][0] * inSignal_f[i][0] - kernel_fft[i][1] * inSignal_f[i][1]) * scale;
        convolution_f[i][1] = (kernel_fft[i][0] * inSignal_f[i][1] + kernel_fft[i][1] * inSignal_f[i][0]) * scale;
    }

    // IFFT convolved signal
    pinv = fftw_plan_dft_1d(padding,
                            convolution_f,
                            convolution,
                            FFTW_BACKWARD,
                            FFTW_ESTIMATE);
    fftw_execute(pinv);


    // DIRECT: compute and compare with expected result
    for (int i = 0; i < padding; i++)
    {
        double expected = 0;

        for (int k = 0; k < padding; k++)
        {
            if((i-k) >=0 && (i-k) < padding){
                expected += input_padding[i-k] * ker_padding[k] ;
            }
        }
        printf("i=%d, FFT: r%f, i%f : Expected: %f\n", i, convolution[i][0], convolution[i][1], expected);
    }

    fftw_plan     pc2r;
    double *out = new double[padding];
    pc2r = fftw_plan_dft_c2r_1d(padding,
                                convolution_f,
                                out,
                                FFTW_ESTIMATE);
    fftw_execute(pc2r);

    for (int i = 0; i < padding; i++)
    {
        printf("i=%d, FFT: %f\n", i, out[i]);
    }
    fftw_destroy_plan(pc2r);


    fftw_destroy_plan(p);
    fftw_destroy_plan(pinv);

    fftw_free(kernel); fftw_free(inSignal); fftw_free(convolution);
    fftw_free(kernel_fft); fftw_free(inSignal_f); fftw_free(convolution_f);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void BenchmarkTest(){

    // Input signal
    const int input_length = 1000;
    double *input = new double[input_length];
    for(int i=0; i<input_length; i++){
        input[i]=rand()%100;  //Generate number between 0 to 99
    }

    // Kernel
    const int kernel_length = 10000;
    double *ker = new double[kernel_length];
    for(int i=0; i<kernel_length; i++){
        ker[i]=rand()%100;  //Generate number between 0 to 99
    }

    // Padding length
    int padding = kernel_length + input_length - 1;

    // Normalization factor
    const double scale = 1./(padding);

    double input_padding[padding];
    double ker_padding[padding];

    // pad input and kernel for direct convolution
    for (int i = 0; i < padding; ++i)
    {
        if (i < kernel_length){
            ker_padding[i] = ker[i];
        }
        else{
            ker_padding[i] = 0;
        }

        if (i < input_length){
            input_padding[i] = input[i];
        }
        else{
            input_padding[i] = 0;
        }
    }


    // FFT STARTS HERE

    // Get starting timepoint
    auto startFFT = high_resolution_clock::now();

    // Declare complex variables
    fftw_complex *kernel, *inSignal, *convolution;
    fftw_complex *kernel_fft, *inSignal_f, *convolution_f;
    fftw_plan     p, pinv;

    // Allocate memory
    kernel = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);
    inSignal    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);
    convolution = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);
    kernel_fft = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);
    inSignal_f    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);
    convolution_f    = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * padding);


    // Signals to be convolved: Complex kernel and input array filling with padding
    for (int i = 0; i < padding; ++i)
    {
        if (i < kernel_length){
            kernel[i][0] = ker[i];
            kernel[i][1] = 0;
        }
        else{
            kernel[i][0] = 0;
            kernel[i][1] = 0;
        }

        if (i < input_length){
            inSignal[i][0] = input[i];
            inSignal[i][1] = 0;
        }
        else{
            inSignal[i][0] = 0;
            inSignal[i][1] = 0;
        }
    }


    // FFT kernel
    p = fftw_plan_dft_1d(padding,
                         kernel,
                         kernel_fft,
                         FFTW_FORWARD,
                         FFTW_ESTIMATE);

    fftw_execute(p);

    // FFT input
    p = fftw_plan_dft_1d(padding,
                         inSignal,
                         inSignal_f,
                         FFTW_FORWARD,
                         FFTW_ESTIMATE);
    fftw_execute(p);

    // convolution in frequency domain
    for(int i = 0; i < padding; ++i)
    {
        convolution_f[i][0] = (kernel_fft[i][0] * inSignal_f[i][0] - kernel_fft[i][1] * inSignal_f[i][1]) * scale;
        convolution_f[i][1] = (kernel_fft[i][0] * inSignal_f[i][1] + kernel_fft[i][1] * inSignal_f[i][0]) * scale;
    }

    // IFFT convolved signal
    pinv = fftw_plan_dft_1d(padding,
                            convolution_f,
                            convolution,
                            FFTW_BACKWARD,
                            FFTW_ESTIMATE);
    fftw_execute(pinv);

    // Get ending timepoint and display time
    auto stopFFT = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stopFFT - startFFT);
    cout << "Time taken by FFT: " << duration.count() << " microseconds" << endl;



    // Get starting timepoint
    auto startDirect = high_resolution_clock::now();

    // DIRECT: compute and compare with expected result
    for (int i = 0; i < padding; i++)
    {
        double expected = 0;

        for (int k = 0; k < padding; k++)
        {
            if((i-k) >=0 && (i-k) < padding){
                expected += input_padding[i-k] * ker_padding[k] ;
            }
        }
        //        printf("i=%d, FFT: r%f, i%f : Expected: %f\n", i, convolution[i][0], convolution[i][1], expected);
    }

    // Get ending timepoint and display time
    auto stopDirect = high_resolution_clock::now();
    auto durationDirect = duration_cast<microseconds>(stopDirect - startDirect);
    cout << "Time taken by Direct method: " << durationDirect.count() << " microseconds" << endl;



    // Get starting timepoint
    auto startDir = high_resolution_clock::now();

    // convert complex convolution_f to real out
    double *out = new double[padding];
    for (int i = 0; i < padding; i++)
    {
        out[i] =  convolution[i][0];
    }

    // Get ending timepoint and display time
    auto stopDir = high_resolution_clock::now();
    auto durationDir = duration_cast<microseconds>(stopDir - startDir);
    cout << "Time taken by foor loop " << durationDir.count() << " microseconds" << endl;


    fftw_destroy_plan(p);
    fftw_destroy_plan(pinv);

    fftw_free(kernel); fftw_free(inSignal); fftw_free(convolution);
    fftw_free(kernel_fft); fftw_free(inSignal_f); fftw_free(convolution_f);

}
