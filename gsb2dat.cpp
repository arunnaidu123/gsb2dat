#include<stdio.h>
#include<stdlib.h>
#include "gsb2dat.hpp"

class gpuVariables gpu;
class hostVariables host;

int gsb2dat();



int main(int argc, char** argv)
{
  gpu.allocate_buufers(); 
  char *fileOut, *r1file, *r2file, *l1file, *l2file, *ts_file, *hdrfile;
  fileOut = new char[500];
  r1file = new char[500];
  r2file = new char[500];
  l1file = new char[500];
  l2file = new char[500];
  hdrfile = new char[500];
  ts_file = new char[500];

  for(int i=0;i<argc;i++)
  {
    if(strcmp(argv[i],"-o")==0)
    {
      i++;
      strcpy(fileOut,argv[i]);
      strcat(fileOut,".dat");
      strcpy(hdrfile,argv[i]);
      strcat(hdrfile,".hdr");
    }
    if(strcmp(argv[i],"-r1")==0)
    {
      i++;
      strcpy(r1file,argv[i]);
    }
    if(strcmp(argv[i],"-r2")==0)
    {
      i++;
      strcpy(r2file,argv[i]);
    }
    if(strcmp(argv[i],"-l1")==0)
    {
      i++;
      strcpy(l1file,argv[i]);
    }
    if(strcmp(argv[i],"-l2")==0)
    {
      i++;
      strcpy(l2file,argv[i]);
    }
    if(strcmp(argv[i],"-t")==0)
    {
      i++;
      strcpy(ts_file,argv[i]);
    }
    if(strcmp(argv[i],"-s")==0)
    {
      i++;
      strcpy(host.source,argv[i]);
    }
    
    if(strcmp(argv[i],"-n")==0)
    {
      i++;
      gpu.nchans = atoi(argv[i]);
      host.nchans = atoi(argv[i]);
    }
    
     if(strcmp(argv[i],"-f")==0)
    {
      i++;
      host.frequency = atof(argv[i]);
    }
    
   

  }

  host.fpr1 = fopen(r1file,"rb");
  if(host.fpr1 == NULL)
  {
    printf("cannot open the required file.... please check the file name \n");
    exit(0);
  }
  
  host.fpr2 = fopen(r2file,"rb");
  if(host.fpr2 == NULL)
  {
    printf("cannot open the required file.... please check the file name \n");
    exit(0);
  }
  
  host.fpl1 = fopen(l1file,"rb");
  if(host.fpl1 == NULL)
  {
    printf("cannot open the required file.... please check the file name \n");
    exit(0);
  }
  host.fpl2 = fopen(l2file,"rb");
  if(host.fpl2 == NULL)
  {
    printf("cannot open the required file.... please check the file name \n");
    exit(0);
  }
  
  host.fpout = fopen(fileOut,"wb");
  if(host.fpout == NULL)
  { 
    printf("cannot open the out put file ...... please check the spce in the location \n");
    exit(0);
  }
  

  int count =0;
  fseek(host.fpr1,0,SEEK_END);
  long file_size = ftell(host.fpr1);
  fseek(host.fpr1,0,SEEK_SET);
  long num_samples = 2*file_size/gpu.nchans;
  
  host.read_ts(ts_file);
  host.write_header(hdrfile,num_samples);
  
  
  while(1)
  {
    std::cout<<"percentage: "<<(100*((long double)count)*64*1024*1024)/((long double)file_size)<<"\r";
    host.fill_pol();
    gsb2dat();
    //transpose_short_data(reinterpret_cast <unsigned short *>(host.dataOut),reinterpret_cast <unsigned short *>(host.dataOut_t));
    //std::cout<<host.data_float[15]<<"\n";
    std::cout<<"percentage: "<<(100*((long double)count)*64*1024*1024)/((long double)file_size)<<"            "<<(int)host.dataOut[15]<<"      \r";
    fwrite(host.dataOut,sizeof(char),256*1024*1024,host.fpout);
    
    count++; 
    std::cout.flush();
  }
  return 0;
}
