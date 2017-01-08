#include<stdio.h>
#include<stdlib.h>
#include<cutil_inline.h>
#include <time.h>
#include <ctime>

//#define N 1000000
#define ThreadsPerBlock 32 
//# define BlockPerGrid 32//2 
#define Bin_Count 64

int size =100;
int BlockPerGrid =2;

// Input Array Variables
int* h_In = NULL;//float* for all input and output array 
int* d_In = NULL;
//int* d_Result =NULL;
//int* h_Result = NULL;
// Output Array
int* h_Out = NULL;
int* d_Out = NULL;


clock_t* atomic_timer =NULL;
clock_t* timer = NULL;
clock_t start_atomic;
clock_t stop_atomic;

//Functions
void Cleanup(void);
void RandomInit(int*, int);//float*
void PrintArray(int*, int);//float*
int iDivUp(int, int );
int CPUhisto(int*,int );
void ParseArguments(int, char**);
//declare timers
unsigned int timer_GPU =0;
unsigned int timer_CPU =0;




// write kernel code here

__global__ void histoAtomic(int * d_PartialHistograms, int * d_Data,unsigned int dataCount,clock_t * atomic_timer ){

    d_PartialHistograms[threadIdx.x]=0;
    const int bid = blockIdx.x;
    clock_t start_atomic, stop_atomic;
 __shared__ unsigned int temp[Bin_Count*ThreadsPerBlock];
   temp[threadIdx.x] = 0;
   __syncthreads();

   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int offset = blockDim.x * gridDim.x;
   while (i < dataCount)
    {
         atomicAdd( &temp[d_Data[i]], 1);    
         i += offset;
    }
    __syncthreads();

   start_atomic = clock();
   atomicAdd( &(d_PartialHistograms[threadIdx.x]), temp[threadIdx.x] );
   stop_atomic= clock();

   if (threadIdx.x ==0){

   atomic_timer[bid]=(stop_atomic-start_atomic);
}

}





//Kernel code
/*__global__ void histoAtomic(int * d_PartialHistograms, int * d_Data,unsigned int dataCount ){
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

  for(int i =0;i< ThreadsPerBlock;i++){
      sum = sum+ s_Hist[threadIdx.x*ThreadsPerBlock];
   }
   d_PartialHistograms[(Bin_Count*blockIdx.x)+threadIdx.x]=sum;
    int sum =0;
   for (int i = 0; i<Bin_Count; i++){
       sum +=s_Hist[threadIdx.x * ThreadsPerBlock];
      // d_PartialHistograms[tid* Bin_Count + i]=s_Hist[threadIdx.x * ThreadsPerBlock];
      // d_PartialHistograms[tid* Bin_Count + i]=sum;
}
       atomicAdd(d_PartialHistograms+threadIdx.x,sum);
}

*/


// write host code here


int main(int argc, char** argv)
{
   ParseArguments(argc, argv);
    int N = size;

   // int N = number;
    printf("Histogram : size %d\n", N);
    size_t in_size = N * sizeof(int);//float
    h_In = (int*)malloc(in_size);//float
   // if (h_In == 0)
     // Cleanup();
   // int* data = RandomInit(h_In, N);//float
      RandomInit(h_In,N);
   

  // printf("\n time here is %f",(double)(clock())/CLOCKS_PER_SEC);
  
 //   for(int i =0; i<N; i++){
   // printf("Input data is %d \n",h_In[i]);
   // }



     //Initialize the timers
    cutilCheckError(cutCreateTimer(&timer_GPU));
    cutilCheckError(cutCreateTimer(&timer_CPU));



     int blocksPerGrid = BlockPerGrid;
     int threads =ThreadsPerBlock ;

    //int sha-sredMemSize = blocksPerGrid * sizeof(float);
   // size_t out_size = blocksPerGrid*Bin_Count * sizeof(float);
    size_t out_size =  64 *sizeof(int);//float//16384
    int histogramCount = iDivUp(N, threads);;
//    size_t partial_size = blocksPerGrid*64*sizeof(int);//float//*histogramCount
   // size_t size1 = threads*blocksPerGrid*sizeof(int);
 // Allocate host output
    h_Out = (int*)malloc(out_size);//float
    timer =(clock_t*)malloc(out_size);
    printf("\nblocks per grid%d\n:",blocksPerGrid);
    printf("Threads per block%d \n:",threads);
    printf("Histogram Count%d \n:" ,histogramCount);

    //allocate memory to device
    cutilSafeCall(cudaMalloc((void**)&d_In,in_size));
    //cutilSafeCall(cudaMalloc((void**)&d_Result,partial_size));
    cutilSafeCall(cudaMalloc((void **)&d_Out,out_size));

    cutilSafeCall(cudaMalloc((void**)&atomic_timer,out_size));
   // cutilSafeCall(cudaMalloc((void **)&d_Out,partial_size));

   // copy data to gpu
   cudaMemcpy(d_In,h_In,in_size,cudaMemcpyHostToDevice);


   // Start GPU timer
   cutilCheckError(cutStartTimer(timer_GPU));


   // invoke kernel
   histoAtomic<<<2*blocksPerGrid,Bin_Count>>>(d_Out,d_In,N,atomic_timer);
   cutilCheckMsg("kernel launch failure");
  // cutilSafeCall( cudaThreadSynchronize() ); // Have host wait for kerne

  //Stop GPU Timer
  cutilCheckError(cutStopTimer(timer_GPU));


  cudaMemcpy(h_Out,d_Out,out_size,cudaMemcpyDeviceToHost);
  cudaMemcpy(timer,atomic_timer,out_size,cudaMemcpyDeviceToHost);

  /* for (int i=1;i<histogramCount;i++){
       for(int j =0;j<64;j++){
           h_Out[j]+=h_Out[i*64+j];
          // printf("Result is %d at %d \n",h_Out[j],j);
}
}
     int r[64];
   for(int i =0;i<64;i++)
{
    r[i] =0;
}
   for(int i =0; i<N; i++){
      r[(int)(h_Out[i])]++;
}*/


//for (int i =0; i<Bin_Count; i++){
//printf("Result is %d\n",h_Out[i]);
//}

/*for(int i =0; i<Bin_Count;i++){
h_Out[0]=h_Out[0]+h_Out[i];
}*/
//printf("GPUR is %d \n",h_Out[0]);



   /*int r[64];
   for(int i =0;i<64;i++)
{
    r[i] =0;
}
   for(int i =0; i<N; i++){
      r[(int)(h_Out[i])]++;
}*/

//printf("GPU Result is %d \n",h);
long histoCount = 0;
for (int i=0; i<64; i++) {
    histoCount += h_Out[i]; }


for(int i =0; i<blocksPerGrid; i++){
printf("\n timer value is %f for block %d",((double)(timer[i])/CLOCKS_PER_SEC),i);
}

printf("\nGPUR is %d \n",histoCount);



   //Start CPU Timer
   cutilCheckError(cutStartTimer(timer_CPU));


    int CPUhistocount = CPUhisto(h_In,N);
    printf("CPU result is %d \n",CPUhistocount);

   // Stop Cpu timer
  cutilCheckError(cutStopTimer(timer_CPU));

//Print Timers
 printf("GPU Execution time: %f (ms)\n",cutGetTimerValue(timer_GPU));
 printf("CPU Execution time : %f(ms)\n",cutGetTimerValue(timer_CPU));



      Cleanup();

}



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
   // if (d_Result)
     //   cudaFree(d_Result);

    // Free host memory
    if (h_In)
        free(h_In);
    if (h_Out)
        free(h_Out);
   // free partial memeory
  // if (h_Result)
    //  free(h_Result);

    //cutilCheckError(cutDeleteTimer(timer_mem));
    //cutilCheckError(cutDeleteTimer(timer_total));
    cutilCheckError(cutDeleteTimer(timer_GPU));
    cutilCheckError(cutDeleteTimer(timer_CPU));
   // cutilCheckError(cutDeleteTimer(timer_CPU_serial));


    cutilSafeCall( cudaThreadExit() );

    exit(0);
}
                  

void RandomInit(int* data, int n)//float
{
    for (int i = 0; i < n; i++){
        data[i] = rand() % 5;
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
       //  printf("CPUresult is %d \n",result[i]);
     }
    // loop to check the print bin count
    // for (int i=0;i<Bin_Count;i++){

   // printf("CPU result is%d \n :",result[i]);
   // } 

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



