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
  
  if(channel==0)
  {
    data_f[sample*1024+512].x = 0.0;
    data_f[sample*1024+512].y = 0.0;
  }
}

__global__ void typeCaste_third(char* dataOut, float2 *data_l, float2 *data_r, float scale)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  dataOut[2*tid] = (char)(int)(data_r[tid].x/(float)scale);;
  dataOut[2*tid+1] = (char)(int)(data_l[tid].x/(float)scale);

}


int gsb2dat()
{
  cudaSetDevice(gpu.device);
  cudaMemcpy(gpu.pol_r_char,host.pol_r_char,128*1024*1024*sizeof(char),cudaMemcpyHostToDevice);
  cudaMemcpy(gpu.pol_l_char,host.pol_l_char,128*1024*1024*sizeof(char),cudaMemcpyHostToDevice);
   
  typeCaste_first<<<64*1024,1024>>>(gpu.pol_r_char, gpu.pol_r);
  typeCaste_first<<<64*1024,1024>>>(gpu.pol_l_char, gpu.pol_l);
  
  for(int i=0;i<1024;i++)
  {
    checkCudaErrors(cufftExecC2C(gpu.plan_512, &gpu.pol_r[i*128*1024], &gpu.pol_r[i*128*1024], CUFFT_INVERSE));
    checkCudaErrors(cufftExecC2C(gpu.plan_512, &gpu.pol_l[i*128*1024], &gpu.pol_l[i*128*1024], CUFFT_INVERSE));
  }
  
  float scale = 32;
  typeCaste_third<<<128*1024,1024>>>(gpu.dataOut, gpu.pol_l,  gpu.pol_r, scale);

  cudaMemcpy(host.dataOut,gpu.dataOut,2*128*1024*1024*sizeof(char),cudaMemcpyDeviceToHost);
  
  return 0;
}


