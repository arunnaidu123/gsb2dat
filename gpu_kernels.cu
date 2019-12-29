#include "gsb2dat.hpp"


__global__ void typeCaste_first(char *data, float2 *data_f)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int sample = tid/512;
  int channel = tid%512;
  char real, imag;
  real = data[tid*2];
  imag = data[tid*2+1];
  
  data_f[sample*1024+channel].x = (float)(int)real;
  data_f[sample*1024+channel].y = (float)(int)imag;  
  
  if(channel>0)
  {
    data_f[sample*1024+1024-channel].x = (float)(int)real;
    data_f[sample*1024+1024-channel].y = -1.0*((float)(int)imag);
  }
  else 
  {
    data_f[sample*1024+512].x = real;
    data_f[sample*1024+512].y = imag;
  }
     
}

__global__ void typeCaste_second(float2 *temp, float2 *data_f)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int sample = tid/(16*1024*1024);
  int channel = tid%(16*1024*1024);
  float real, imag;
  real = data_f[sample*(16*1024*1024)+channel].x;
  imag = data_f[sample*(16*1024*1024)+channel].y;
  temp[sample*(32*1024*1024)+channel].x = real/262144.0;
  temp[sample*(32*1024*1024)+channel].y = imag/262144.0;

  if(tid>0)
  {
    temp[sample*(32*1024*1024)+(32*1024*1024)-channel].x =  real/262144.0;
    temp[sample*(32*1024*1024)+(32*1024*1024)-channel].y = -1.0*imag/262144.0;
  }
  else
  {
    temp[sample*(32*1024*1024)+(16*1024*1024)].x = real;
    temp[sample*(32*1024*1024)+(16*1024*1024)].y = imag;
  }
}


__global__ void typeCaste_place_holder(float *dataOut, float2 *data_l, float2 *data_r, float scale, int nchans)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nfft = nchans/2;
  int n = nfft/2;
  int sample = tid/(n);
  int channel = tid%(n);
  
  float real,imag;
  
  for(int i=0; i<4; i++)
  {
    real = (data_r[i*32*1024*1024+sample*nfft+channel].x/scale);
    imag = (data_r[i*32*1024*1024+sample*nfft+channel].y/scale);

    dataOut[sample*nchans*4+16*channel+4*i] = real;
    dataOut[sample*nchans*4+16*channel+4*i+1] = imag;

    real = (data_l[3*32*1024*1024+sample*nfft+channel].x/scale);
    imag = (data_l[3*32*1024*1024+sample*nfft+channel].y/scale);

    dataOut[sample*nchans*4+16*channel+4*i+2] = real;
    dataOut[sample*nchans*4+16*channel+4*i+3] = imag;
  }


}

__global__ void typeCaste_third(char* dataOut, float2 *data_l, float2 *data_r, float scale, int nchans)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  int sample = tid/(nchans);
  int channel = tid%(nchans);

  float real, imag;
  
  real = (char)(int)(data_l[sample*nchans*2+channel].x/scale);
  imag = (char)(int)(data_l[sample*nchans*2+channel].y/scale);

  dataOut[sample*nchans*4+channel*4+0] = real;
  dataOut[sample*nchans*4+channel*4+1] = imag;

  real = (char)(int)(data_r[sample*nchans*2+channel].x/scale);
  imag = (char)(int)(data_r[sample*nchans*2+channel].y/scale);

  dataOut[sample*nchans*4+channel*4+2] = real;
  dataOut[sample*nchans*4+channel*4+3] = imag;
}

__global__ void typeCaste_test(char *dataOut,  char *data_r, char *data_l)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  dataOut[4*tid] = data_r[2*tid];
  dataOut[4*tid+1] = data_r[2*tid+1];
  dataOut[4*tid+2] = data_l[2*tid];
  dataOut[4*tid+3] = data_l[2*tid+1];
}

int gsb2dat()
{
  cudaSetDevice(gpu.device);
  cudaMemcpy(gpu.pol_r_char,host.pol_r_char,128*1024*1024*sizeof(char),cudaMemcpyHostToDevice);
  cudaMemcpy(gpu.pol_l_char,host.pol_l_char,128*1024*1024*sizeof(char),cudaMemcpyHostToDevice);
   
  //typeCaste_test<<<64*1024,1024>>>(gpu.dataOut, gpu.pol_r_char, gpu.pol_l_char);
  
  typeCaste_first<<<64*1024,1024>>>(gpu.pol_r_char, gpu.pol_r);
  typeCaste_first<<<64*1024,1024>>>(gpu.pol_l_char, gpu.pol_l);
  
  for(int i=0;i<1024;i++)
  {
    checkCudaErrors(cufftExecC2C(gpu.plan_512, &gpu.pol_r[i*128*1024], &gpu.pol_r[i*128*1024], CUFFT_INVERSE));
    checkCudaErrors(cufftExecC2C(gpu.plan_512, &gpu.pol_l[i*128*1024], &gpu.pol_l[i*128*1024], CUFFT_INVERSE));
  }
  
  //checkCudaErrors(cufftExecC2C(gpu.plan_f, gpu.pol_l, gpu.pol_l, CUFFT_FORWARD));
  //checkCudaErrors(cufftExecC2C(gpu.plan_f, gpu.pol_r, gpu.pol_r, CUFFT_FORWARD));
  
   /* 
  typeCaste_second<<<64*1024,1024>>>(gpu.temp, gpu.pol_r);
  checkCudaErrors(cufftExecC2C(gpu.plan_b, gpu.temp, gpu.pol_r, CUFFT_INVERSE));
  typeCaste_second<<<64*1024,1024>>>(gpu.temp, gpu.pol_l);
  checkCudaErrors(cufftExecC2C(gpu.plan_b, gpu.temp, gpu.pol_l, CUFFT_INVERSE));
  */
  for(int i=0;i<1024;i++)
  {
    checkCudaErrors(cufftExecC2C(gpu.plan_nchans, &gpu.pol_r[i*128*1024], &gpu.pol_r[i*128*1024], CUFFT_FORWARD));
    checkCudaErrors(cufftExecC2C(gpu.plan_nchans, &gpu.pol_l[i*128*1024], &gpu.pol_l[i*128*1024], CUFFT_FORWARD));
  }
  

  float scale = 32*sqrt((float)2*host.nchans);
  typeCaste_third<<<64*1024,1024>>>(gpu.dataOut, gpu.pol_l,  gpu.pol_r, scale, gpu.nchans);

  cudaMemcpy(host.dataOut,gpu.dataOut,2*128*1024*1024*sizeof(char),cudaMemcpyDeviceToHost);
  
  return 0; 
}


