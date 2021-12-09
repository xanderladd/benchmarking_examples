
#include "Util.h"
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[])
{
	int num_devices;
    int stim_ind;
    int globalRank;
    int * devGlobalRank;
	cudaGetDeviceCount(&num_devices);
	//num_devices = 1;
	for (int i = 0; i < num_devices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//printf("Device Number: %d\n", i);
		//printf("  Device name: %s\n", prop.name);
		//printf("  Memory Clock Rate (KHz): %d\n",		prop.memoryClockRate);
		//printf("  Memory Bus Width (bits): %d\n",	prop.memoryBusWidth);
		//printf("  Peak Memory Bandwidth (GB/s): %f\n\n",		2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}
	//RunByModelSerial();// FOR ROY!!!!!!!!!!!!!!! // Run, output VHot and look
    
    if (argc>1){
        stim_ind = char2int(argv[1]); 
        }
        else{
        stim_ind = 0;
        }
     if (argc>2){
         globalRank = char2int(argv[2]); 
         cudaMalloc((void**)&devGlobalRank, sizeof(int));
         cudaMemcpy(devGlobalRank, &globalRank, sizeof(int), cudaMemcpyHostToDevice);
        }
    printf("global rank is %d \n",globalRank );
    printf("NUM DEVICES %d", num_devices);
    printf("got stim num %d\n", stim_ind); 
    printf("using dev %d",stim_ind % num_devices );

	CUDA_RT_CALL(cudaSetDevice(stim_ind % num_devices));
	RunByModelP(argc, stim_ind, globalRank );
	return 0;
}

