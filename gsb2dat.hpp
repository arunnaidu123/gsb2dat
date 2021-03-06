#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftw.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdio.h>

#define size 4194304
#define channels 512

class gpuVariables 
{
  public: 
  char *pol_r_char;
  char *pol_l_char;
  float2 *pol_r;
  float2 *pol_l;
  char *dataOut;
  cufftHandle plan_512;
  int device = 1;
  int allocate_buffers(int nc, int d)
  {
    
    cudaSetDevice(d);
    cudaGetDevice(&device);
    std::cout<<"device: "<<device<<"\n";

    int batch = 128;
    int rank = 1;
    int nRows = 1024;
    int n[1] = {nRows};
    int idist = 1024;
    int odist = 1024;
    int inembed[] = {0};
    int onembed[] = {0};
    int istride = 1;
    int ostride = 1;
    
    checkCudaErrors(cufftPlanMany(&plan_512, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch)); 
    
    checkCudaErrors(cudaMalloc((void **)&pol_r, sizeof(float2)*128*1024*1024));
    checkCudaErrors(cudaMalloc((void **)&pol_l, sizeof(float2)*128*1024*1024));
    checkCudaErrors(cudaMalloc((void **)&dataOut, sizeof(char)*2*128*1024*1024));
    checkCudaErrors(cudaMalloc((void **)&pol_r_char, sizeof(char)*128*1024*1024));
    checkCudaErrors(cudaMalloc((void **)&pol_l_char, sizeof(char)*128*1024*1024));
    return 0;
  }
  
  ~gpuVariables()
  { 
    cudaSetDevice(device);
    cufftDestroy(plan_512);
    cudaFree(pol_l);
    cudaFree(pol_r);
    cudaFree(dataOut);
    cudaFree(pol_r_char);
    cudaFree(pol_l_char);
  }
};

class hostVariables 
{
  public:
  char *data_r1, *data_r2, *data_l1, *data_l2;
  char *pol_r_char;
  char *pol_l_char;
  FILE *fpl1, *fpl2, *fpr1, *fpr2, *fpout, *fpout_l;
  char *dataOut;
  char* dataOut_t;
  char  *timestamp, *source;
  long picoseconds;
  int nchans;
  float frequency;
  hostVariables()
  {
    data_r1 = (char*) malloc(sizeof(char)*64*1024*1024);
    data_r2 = (char*) malloc(sizeof(char)*64*1024*1024);  
    data_l1 = (char*) malloc(sizeof(char)*64*1024*1024);
    data_l2 = (char*) malloc(sizeof(char)*64*1024*1024);
    dataOut_t = (char*) malloc(sizeof(char)*256*1024*1024);
    dataOut = (char*) malloc(sizeof(char)*256*1024*1024);
    pol_r_char = (char*) malloc(sizeof(char)*128*1024*1024);
    pol_l_char = (char*) malloc(sizeof(char)*128*1024*1024);

    timestamp = (char*) malloc(sizeof(char)*100);
    source = (char*) malloc(sizeof(char)*100);
  }
  
  void fill_pol()
  {
    int nread1 = fread(data_r1,1,64*1024*1024,fpr1);
    int nread2 = fread(data_r2,1,64*1024*1024,fpr2);
    if(nread1 != 64*1024*1024 && nread2 != 64*1024*1024)
    {
      printf("END of the file exit \n");
      close_all();
    }

    
    nread1 = fread(data_l1,1,64*1024*1024,fpl1);
    nread2 = fread(data_l2,1,64*1024*1024,fpl2);
    if(nread1 != 64*1024*1024 && nread2 != 64*1024*1024)
    {
      printf("END of the file exit \n");
      close_all();
    }
    
    for(int i=0;i<16;i++)
    {
      memcpy(&pol_r_char[2*i*size],&data_r1[size*i],size);
      memcpy(&pol_r_char[(2*i+1)*size],&data_r2[size*i],size);
      memcpy(&pol_l_char[2*i*size],&data_l1[size*i],size);
      memcpy(&pol_l_char[(2*i+1)*size],&data_l2[size*i],size);
      std::cout.flush();
    }
    for(int i=0;i<64*1024*1024;i++)
    {
      dataOut_t[4*i] = pol_r_char[2*i];
      dataOut_t[4*i+1] = pol_r_char[2*i+1];
      dataOut_t[4*i+2] = pol_l_char[2*i];
      dataOut_t[4*i+3] = pol_l_char[2*i+1];
    }
    //fwrite(dataOut_t,sizeof(char),256*1024*1024,fpout_l);
  }
  
  int read_ts(char* file_name)
  {
    FILE *fpts;
    int y1,y2,m1,m2,d1,d2,h1,h2,mm1,mm2,s1,s2,t1,t2;
    double f1,f2;
    fpts = fopen(file_name,"r");
    if(fpts==NULL)
    {
      std::cout<<"Cannot open the timestamp file \n";
      close_all();
      exit(0);
    }

    fscanf(fpts,"%d %d %d %d %d %d %lf %d %d %d %d %d %d %lf %d %d ",&y1,&m1,&d1,&h1,&mm1,&s1,&f1,&y2,&m2,&d2,&h2,&mm2,&s2,&f2,&t1,&t2);
    sprintf(timestamp,"%d-%02d-%02d-%02d:%02d:%02d",y2,m2,d2,h2,mm2,s2);
    picoseconds = (long)(f2*1e12);
    fclose(fpts);
    return 0;
  }
  
  int write_header(char* file_name,long num_samples)
  {
    FILE *fphdr;

    fphdr = fopen(file_name,"w");
    if(fphdr==NULL)
    {
      std::cout<<"Cannot write the header file \n";
      close_all();
      exit(0);
    }
    
    fprintf(fphdr,"HDR_VERSION 1.0 \n");
    fprintf(fphdr,"BW    16.6666666 \n");
    fprintf(fphdr,"FREQ %lf \n",((double)frequency+8.33333333));
    fprintf(fphdr,"TELESCOPE  GMRT \n");
    fprintf(fphdr,"RECEIVER   GSB \n");
    fprintf(fphdr,"INSTRUMENT GSB \n");
    fprintf(fphdr,"SOURCE %s \n",source);
    fprintf(fphdr,"MODE PSR \n");
    fprintf(fphdr,"NBIT 8 \n");
    fprintf(fphdr,"NCHAN 1 \n");
    fprintf(fphdr,"NDIM 1 \n");
    fprintf(fphdr,"NPOL 2 \n");
    fprintf(fphdr,"NDAT %ld \n",num_samples);
    fprintf(fphdr,"OBS_OFFSET 0 \n");
    fprintf(fphdr,"UTC_START %s\n",timestamp);
    fprintf(fphdr,"PICOSECONDS %ld\n",picoseconds);
    fprintf(fphdr,"TSAMP %1.10lf\n",0.030000036);
    fprintf(fphdr,"RESOLUTION 2\n");
    //fprintf(fphdr,"RA %s \n",ra_char);
    //fprintf(fphdr,"DEC %s \n",dec_char);
    fclose(fphdr);
    return 0;
  }
  int close_all()
  {
    fclose(fpr1);
    fclose(fpr2);
    fclose(fpl1);
    fclose(fpl2);
    fclose(fpout);
    fclose(fpout_l);
    exit(0);
    return 0;
  }

};

extern class gpuVariables gpu;
extern class hostVariables host;

