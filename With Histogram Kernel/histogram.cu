
#include<stdio.h>
#include<stdlib.h>
#include<cutil_inline.h>

//#define N 1000000
#define ThreadsPerBlock 32 
//# define BlockPerGrid 1 
#define Bin_Count 64
#define MergeThreadBlocksize 32
// variables

int size =100;
int BlockPerGrid=1;
//int ThreadsPerBlock =32;

// Input Array Variables
int* h_In = NULL;//float* for all input and output array 
int* d_In = NULL;
int* d_Result =NULL;
int* h_Result = NULL;
// Output Array
int* h_Out = NULL;
int* d_Out = NULL;

//Functions
void Cleanup(void);
void RandomInit(int*, int);//float*
void PrintArray(int*, int);//float*
int iDivUp(int, int );
int CPUhisto(int*,int );
void ParseArguments(int, char**);


//timer declarations
unsigned int timer_CPU =0;
unsigned int timer_GPU = 0;
unsigned int timer_Partial_Sum = 0;
unsigned int timer_total = 0;
unsigned int timer_mem =0;


//Kernel code
__global__ void histogram64Kernel(int * d_PartialHistograms, int * d_Data,unsigned int dataCount ){
   __shared__ int s_Hist[ThreadsPerBlock * Bin_Count];
   int tid = threadIdx.x +blockIdx.x *blockDim.x;

 //   for (int i = 0; i<Bin_Count; i++)
   //    d_PartialHistograms[tid* Bin_Count + i]=0;

   

   for (int i =0; i<Bin_Count;i++){
       s_Hist[threadIdx.x *ThreadsPerBlock+i]=0;
   }
   int THREAD_N = blockDim.x * gridDim.x;
   for(int pos = tid; pos< dataCount; pos = pos + THREAD_N){
      int data = d_Data[pos];
      s_Hist[data + threadIdx.x * Bin_Count]+= 1;
   }
   __syncthreads();

  // unsigned int sum =0;
   
/*  for(int i =0;i< ThreadsPerBlock;i++){
      sum = sum+ s_Hist[threadIdx.x*ThreadsPerBlock];
   }
   d_PartialHistograms[(Bin_Count*blockIdx.x)+threadIdx.x]=sum;*/
   
   for (int i = 0; i<Bin_Count; i++)
       d_PartialHistograms[tid* Bin_Count + i]=s_Hist[threadIdx.x * ThreadsPerBlock];
}

/////////////////////////////////////////////////////////////////////////////////////



//Host code
int main(int argc, char** argv)
{
   
   ParseArguments(argc, argv);
    int N = size;
    int blocksPerGrid = BlockPerGrid;
    int threads =ThreadsPerBlock ;


    printf("Histogram : size %d\n", N);
    size_t in_size = N * sizeof(int);//float
    h_In = (int*)malloc(in_size);//float
   // if (h_In == 0)
     // Cleanup();
   // int* data = RandomInit(h_In, N);//float
      RandomInit(h_In,N);
   // print data for checking
  /*  for(int i =0; i<N; i++){

    printf("Data at location %d is %d \n ", i , h_In[i]);
    }*/  


   // int blocksPerGrid = blocks;


  //   int blocksPerGrid = BlockPerGrid;
   //  int threads =ThreadsPerBlock ;

    //int sha-sredMemSize = blocksPerGrid * sizeof(float);
   // size_t out_size = blocksPerGrid*Bin_Count * sizeof(float);
   size_t out_size = 16384 * 64 *sizeof(int);//float//16384
    int histogramCount = iDivUp(N, threads);;
    size_t partial_size = histogramCount*64*sizeof(int);//float
   // size_t size1 = threads*blocksPerGrid*sizeof(int);
 // Allocate host output
    h_Out = (int*)malloc(out_size);//float
  //  if (h_Out == 0)
    //  Cleanup();
    // print calculated factors


    h_Result = (int*)malloc(partial_size);//float
   // if (h_Result == 0)
     // Cleanup();


   // intialize the timers
   cutilCheckError(cutCreateTimer(&timer_CPU));
   cutilCheckError(cutCreateTimer(&timer_GPU));

   cutilCheckError(cutCreateTimer(&timer_Partial_Sum));
   cutilCheckError(cutCreateTimer(&timer_total));
   cutilCheckError(cutCreateTimer(&timer_mem));

   




    printf("blocks per grid%d\n:",blocksPerGrid);
    printf("Threads per block%d \n:",threads);
    printf("Histogram Count%d \n:" ,histogramCount);

    //allocate memory to device
    cutilSafeCall(cudaMalloc((void**)&d_In,in_size));
    cutilSafeCall(cudaMalloc((void**)&d_Result,partial_size));
    cutilSafeCall(cudaMalloc((void **)&d_Out,out_size));
   // cutilSafeCall(cudaMalloc((void **)&d_Out,partial_size));

   cutilCheckError(cutStartTimer(timer_total));
   cutilCheckError(cutStartTimer(timer_mem));
   // copy data to gpu
   cudaMemcpy(d_In,h_In,in_size,cudaMemcpyHostToDevice);

  cutilCheckError(cutStopTimer(timer_mem));
  printf("CPU to GPU transfer time %f(ms)\n",cutGetTimerValue(timer_mem));
  

   //start GPU timer
  cutilCheckError(cutStartTimer(timer_GPU));

//  cutilCheckError(cutStartTimer(timer_Overall_GPU));

   // invoke kernel
   histogram64Kernel<<<histogramCount,threads>>>(d_Result,d_In,N);
   cutilCheckMsg("kernel launch failure");
  // cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel
   

  //stop GPU TIMER
 cutilCheckError(cutStopTimer(timer_GPU));

   cutilCheckError(cutCreateTimer(&timer_mem));
   cutilCheckError(cutStartTimer(timer_mem));

   cudaMemcpy(h_Result,d_Result,partial_size,cudaMemcpyDeviceToHost);
   cutilCheckError(cutStopTimer(timer_mem));
   printf("GPU to CPU transfer time %f(ms)\n",cutGetTimerValue(timer_mem));

  //Copy results of generate kernel to host
  //cudaMemcpy(d_Result,h_Result,partial_size,cudaMemcpyDeviceToHost);
   
 /* for(int i =0; i<N; i++){
      printf("data back to memory is%f\nat %d\n",h_Result[i],i);
  }*/
  //invoke MergeKernel
 //  mergeHistogram64Kernel<<<Bin_Count,MergeThreadBlocksize>>>(d_Out,d_Result,histogramCount);
  // imergeHistogram64Kernel<<<1,N>>>(d_Out,d_Result,histogramCount);
  // cutilCheckMsg("kernel launch failure");
  // cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kernel

   //get the results from Merge Kernel
 //  cudaMemcpy(h_Out,d_Out,out_size,cudaMemcpyDeviceToHost);
  /* for(int i =0; i<N; i++){
   //   h_Result [0]= h_Result[0]+h_Result[i];
       printf("Result is %d\n",h_Result[i]);
}*/
  // int Final_Block = min(32,histogramCount);

   
   cutilCheckError(cutStartTimer(timer_Partial_Sum));

   for (int i=1;i<histogramCount;i++){
       for(int j =0;j<64;j++){
           h_Result[j]+=h_Result[i*64+j];
          // printf("Result is %d at %d \n",h_Out[j],j);
}
}

   // for(int i =0;i<64;i++){
    // printf("result is %d \n",h_Out[i]);
//}
  // printf("%d\n",h_Result[0]);
  
   // Sum at the CPU
 /*  float sum = 0;
   for (int i =0; i<Bin_Count; i++)
   {
      sum = sum+ h_Out[i];
   }
   printf("sum is %f\n",sum);*/

 /* for(int i =0; i<histogramCount;i++){

     for(int j =0;j<Bin_Count;j++){

       h_Out[j]=h_out[j]+h_Out[i*Bin_Count+j];
     }*/

// cpu sum
   int r[64];
   for(int i =0;i<64;i++)
{
    r[i] =0;
}
   for(int i =0; i<N; i++){
      r[(int)(h_Result[i])]++;
}

 cutilCheckError(cutStopTimer(timer_Partial_Sum));
  cutilCheckError(cutStopTimer(timer_total));

//  for(int i =0; i<64;i++){

 //  r[0]=r[0]+r[i];
//  printf("GPUR is %d\n",r[i]);
//}

 // printf("GPU Resut is: %d\n",r[0]);


   // STart CPU Timer
    cutilCheckError(cutStartTimer(timer_CPU));

   int CPUhistocount = CPUhisto(h_In,N);
   printf("CPU result is %d \n",CPUhistocount);

  //Stop CPU timer
  cutilCheckError(cutStopTimer(timer_CPU));



// Print Timer results
   printf("GPU Execution time %f(ms)\n",cutGetTimerValue(timer_GPU));
   printf("CPU  Partial Sum Execution time %f(ms)\n",cutGetTimerValue(timer_Partial_Sum));
   printf("Total Execution time %f(ms)\n",cutGetTimerValue(timer_total));

   printf("CPU Execution time %f(ms)\n",cutGetTimerValue(timer_CPU));


/*    int result[64];
    for (int i=0; i<64; i++)
        result [i] = 0;
    // --- Histogram calculation on the host
    for (int i=0; i<N; i++){
         result[(int)(h_In[i])]++;
        // printf("CPUresult is %d \n",result[i]);
     }
    for(int i =0;i<64;i++){
       result[0]=result[0]+result[i];
     }
       printf("CPUR is %d \n",result[0]);
   // }*/
       printf("GPUR is %d \n",r[0]);


 /*  for(int i =0; i< Bin_Count; i++){
      printf("result is %ffor %d\n",h_Out[i],i);
}*/

/*   for(int i =1; i<histogramCount; i++){
       for(int j =0; j<Bin_Count; j++){
           h_Out[0]+=h_Out[i*Bin_Count+j];
}


}*/
//printf("r%f\n",h_Out[0]);
   Cleanup();
  
}
  /* or(int i = 1; i < FINAL_BLOCK_N; i++){

            for(int j = 0; j < BIN_COUNT; j++)

                h_Result64GPU[j] += h_Result64GPU[i * BIN_COUNT + j];

        }

        for(int i = 0; i < BIN_COUNT; i++)

            h_Result[i] = h_Result64GPU[i];
*/
     

int iDivUp(int a, int b){

    return (a % b != 0) ? (a / b + 1) : (a / b);

}


void Cleanup(void)
{
    // Free device memory
    if (d_In)
        cudaFree(d_In);
    if (d_Out)
        cudaFree(d_Out);
    if (d_Result)
        cudaFree(d_Result);

    // Free host memory
    if (h_In)
        free(h_In);
    if (h_Out)
        free(h_Out);
   // free partial memeory
   if (h_Result)
      free(h_Result);

    //cutilCheckError(cutDeleteTimer(timer_mem));
    //cutilCheckError(cutDeleteTimer(timer_total));
    cutilCheckError(cutDeleteTimer(timer_GPU));
    cutilCheckError(cutDeleteTimer(timer_GPU));
    cutilCheckError(cutDeleteTimer(timer_mem));
    cutilCheckError(cutDeleteTimer(timer_Partial_Sum));
    cutilCheckError(cutDeleteTimer(timer_total));
    cutilSafeCall( cudaThreadExit() );

    exit(0);
}

void RandomInit(int* data, int n)//float
{
    for (int i = 0; i < n; i++){
        data[i] = rand() % 64;
       // printf("data is %d\n",data[i]);
    }
   // return data;
}

void PrintArray(int* data, int n)//float
{
    for (int i = 0; i < n; i++)
        printf("[%d] => %f\n",i,data[i]);
}

int CPUhisto(int* data,int n)
{
   int result[64];
    for (int i=0; i<64; i++)
        result [i] = 0;
    // --- Histogram calculation on the host
    for (int i=0; i<n; i++){
         result[(int)(data[i])]++;
   //      printf("result is %f \n",result[i]);
     }
    // loop to check the print bin count
    /* for (int i=0;i<20;i++){

    printf("result is%f for i%d\n :",result[i],i);
    } */

    int sum =0;
    for(int i =0; i<n;i++){
    sum =sum+ result[i];
   // printf("sum is %d\n",sum); 
    
    }
    long histoCount = 0;
    for (int i=0; i<64; i++) {
        histoCount += result[i]; }
   // printf("Histogram Sum: %ld\n", histoCount);
   return histoCount;
}





// Parse program arguments
void ParseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 || strcmp(argv[i], "-size") == 0) {
                  size = atoi(argv[i+1]);
                  i = i + 1;
        }
         if (strcmp(argv[i], "--blocks") == 0 || strcmp(argv[i], "-blocks") == 0) {
                  BlockPerGrid = atoi(argv[i+1]);
                  i = i + 1;
        }

    }
}

