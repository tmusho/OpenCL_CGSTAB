/* OpenCL Inteface for Fortran90*/
/* Written by Terence Musho - West Virginia University*/

/*Copyright (c) 2016 Dr. Terence Musho - West Virginia University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdlib.h>
#include <stdio.h>
#include <CL/opencl.h>
#include <string.h>
#include <sys/stat.h>

extern "C" {
// globals
cl_platform_id gPlatform[2];
cl_context     gContext;
cl_device_id   gDevice_id;
cl_event       gEvents[100]; //max of 100 events
cl_uint        gNumEvents=0;

cl_program clcgstab; //conjugent gradient Opencl Program
cl_program clreduce; //reduction routine kernels
cl_program clcgstab1D; //conjugent gradient Opencl Program - 1D Version

cl_kernel clcgstab_kernel;
cl_kernel clcgstab_reduce;
cl_kernel clcgstab_reduce_abs;
cl_kernel clcgstab_zer; 
cl_kernel clcgstab_irv; 
cl_kernel clcgstab_pmd;
cl_kernel clcgstab_pmd2;  
cl_kernel clcgstab_iwa; 
cl_kernel clcgstab_cbo; 
cl_kernel clcgstab_cpk; 
cl_kernel clcgstab_sfs; 
cl_kernel clcgstab_sfs2; 
cl_kernel clcgstab_sbs; 
cl_kernel clcgstab_sbs2; 
cl_kernel clcgstab_cuk; 
cl_kernel clcgstab_ruk; 
cl_kernel clcgstab_ufr; 
cl_kernel clcgstab_smy; 
cl_kernel clcgstab_smy2; 
cl_kernel clcgstab_cav; 
cl_kernel clcgstab_caf; 
cl_kernel clcgstab_cfr;

cl_kernel clreduce_1d;
cl_kernel clreduce_abs_1d;

cl_kernel clcgstab1D_kernel;
cl_kernel clcgstab1D_reduce;
cl_kernel clcgstab1D_reduce_abs;
cl_kernel clcgstab1D_zer; 
cl_kernel clcgstab1D_irv; 
cl_kernel clcgstab1D_pmd;
cl_kernel clcgstab1D_pmd2;  
cl_kernel clcgstab1D_iwa; 
cl_kernel clcgstab1D_cbo; 
cl_kernel clcgstab1D_cpk; 
cl_kernel clcgstab1D_sfs; 
cl_kernel clcgstab1D_sfs2; 
cl_kernel clcgstab1D_sbs; 
cl_kernel clcgstab1D_sbs2; 
cl_kernel clcgstab1D_cuk; 
cl_kernel clcgstab1D_ruk; 
cl_kernel clcgstab1D_ufr; 
cl_kernel clcgstab1D_smy; 
cl_kernel clcgstab1D_smy2; 
cl_kernel clcgstab1D_cav; 
cl_kernel clcgstab1D_caf; 
cl_kernel clcgstab1D_cfr;
cl_kernel clcgstab1D_sfs_init;

cl_kernel kernel_list[47];


cl_event events[2];

//fortran appends a underscore to the function-name
#define clinterface_init clinterface_init_
#define clinterface_get_devices clinterface_get_devices_
#define clinterface_queue_create clinterface_queue_create_
#define clinterface_malloc clinterface_malloc_
#define clinterface_malloc_readonly clinterface_malloc_readonly_
#define clinterface_malloc_writeonly clinterface_malloc_writeonly_
#define clinterface_clflush clinterface_clflush_
#define clinterface_zsetvector clinterface_zsetvector_
#define clinterface_rtinit clinterface_rtinit_
#define clinterface_zsetmatrix clinterface_zsetmatrix_
#define clinterface_zsetvector clinterface_zsetvector_
#define clinterface_zgetmatrix clinterface_zgetmatrix_
#define clinterface_buildkernels clinterface_buildkernels_
#define clinterface_zsetmatrix_mapped clinterface_zsetmatrix_mapped_
#define clinterface_malloc_mapped clinterface_malloc_mapped_
#define clinterface_malloc_mapped_readonly clinterface_malloc_mapped_readonly_
#define clinterface_malloc_mapped_writeonly clinterface_malloc_mapped_writeonly_
#define clinterface_clsetkernelarg clinterface_clsetkernelarg_
#define clinterface_clsetkernelarg_ptr clinterface_clsetkernelarg_ptr_
#define clinterface_clsetkernelarg_shmem clinterface_clsetkernelarg_shmem_
#define clinterface_clenqueuendrangekernel clinterface_clenqueuendrangekernel_
#define clinterface_isetmatrix clinterface_isetmatrix_
#define clinterface_isetmatrix_nb clinterface_isetmatrix_nb_
#define clinterface_isetvector clinterface_isetvector_
#define clinterface_igetmatrix clinterface_igetmatrix_
#define clinterface_igetmatrix_nb clinterface_igetmatrix_nb_
#define clinterface_igetvector clinterface_igetvector_
#define clinterface_dsetmatrix clinterface_dsetmatrix_
#define clinterface_dsetmatrix_nb clinterface_dsetmatrix_nb_
#define clinterface_dsetvector clinterface_dsetvector_
#define clinterface_dgetmatrix clinterface_dgetmatrix_
#define clinterface_dgetmatrix_nb clinterface_dgetmatrix_nb_
#define clinterface_dgetvector clinterface_dgetvector_
#define clinterface_clretainmemobject clinterface_clretainmemobject_
#define clinterface_clreleasememobject clinterface_clreleasememobject_
#define clinterface_malloc_pinned clinterface_malloc_pinned_
#define clinterface_malloc_pinned_readonly clinterface_malloc_pinned_readonly_
#define clinterface_malloc_pinned_writeonly clinterface_malloc_pinned_writeonly_
#define clinterface_clbarrier clinterface_clbarrier_
#define clinterface_queue_create_async clinterface_queue_create_async_
#define clinterface_clfinish clinterface_clfinish_

// context error handle
void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
	fprintf(stderr, "OpenCL Error (via pfn_notify): %s\n", errinfo);
}

// initialization
static char *
FileToString(const char *fileName) {

   size_t numBytes, numRead;
   struct stat fileInfo;
   char *contents;
   FILE *file;

   if (stat(fileName, &fileInfo) != 0) {
      printf("FileToString: Unable to stat %s\n",
              fileName);
      return NULL;
   }

   if ((file = fopen(fileName, "rb")) == NULL) {
      printf("FileToString: Unable to open %s\n",
              fileName);
      return NULL;
   }

   numBytes = fileInfo.st_size + 1;
   if ((contents = (char *) malloc(numBytes)) == NULL) {
      printf("FileToString: Unable to allocate %u bytes!\n", numBytes);
      return NULL;
   }

   if ((numRead = fread(contents,
                        1, fileInfo.st_size, file)) != fileInfo.st_size) {
      printf("FileToString: Expected %u bytes, but only read %u!\n",
              numBytes, numRead);
   }
   contents[numRead] = '\0';
   return contents;
}

cl_int clinterface_init(cl_int *rank)
{
    cl_uint num;
    cl_int err;
    cl_device_id devices[10];
    cl_uint plat;
    char buffer[10240];
    char buffer2[10240];

    //on my system the first platform is cpu and second is gpu
    plat=0; //0=cpu 1=gpu
    err = clGetPlatformIDs( 2, gPlatform, NULL );
    if(err != 0) return err;

    err = clGetDeviceIDs( gPlatform[plat], CL_DEVICE_TYPE_GPU, 10, devices, &num );
    if(err != 0) return err;
    printf("\tAvailble Number of GPU Devices = %d\n",num);
    fflush(stdout);
    printf("\t--------------------------------------------------------------\n");
    for(int i=0; i<num; i++){
      gDevice_id = devices[i];
      //err = clGetDeviceInfo(gDevice_id,CL_DEVICE_VENDOR_ID,sizeof(buffer),(void *)buffer,NULL);
      err = clGetDeviceInfo(gDevice_id,CL_DEVICE_NAME,sizeof(buffer2),(void *)buffer2,NULL);
      printf("\t  Device %d - %s\n",i+1,buffer2);
    }
    printf("\t--------------------------------------------------------------\n");

    cl_context_properties properties[3] =
        { CL_CONTEXT_PLATFORM, (cl_context_properties) gPlatform[plat], 0 };
    //num = *rank+1; //force one gpu edit:tmusho
    num = *rank; //force one gpu edit:tmusho
    gDevice_id = devices[*rank];
    //err = clGetDeviceInfo(gDevice_id,CL_DEVICE_VENDOR_ID,sizeof(buffer),(void *)buffer,NULL);

    if(err != 0) return err;
    err = clGetDeviceInfo(gDevice_id,CL_DEVICE_NAME,sizeof(buffer2),(void *)buffer2,NULL);
    if(err != 0) return err;

    printf("\n\tProc Rank to GPU Association\n");
    printf("\t--------------------------------------------------------------\n");
    printf("\t  Proc Rank %d -> Device Number %d - %s\n",*rank,num,buffer2);
    printf("\t--------------------------------------------------------------\n");
    gContext = clCreateContext( properties, 1, &devices[*rank], &pfn_notify, NULL, &err );
    if(err != 0) return err;
    
    fflush(stdout);
    //err = clAmdBlasSetup();
    //if(err != 0) return err;

    //clinterface_rtinit();

    return err;
}

cl_int clinterface_clbuild(cl_context context, cl_device_id* device_id, cl_program* program, char* prog_name)
{
  cl_int err;
  const char *source = FileToString(prog_name);
  size_t sourceSize[] = { strlen(source) };
  *program = clCreateProgramWithSource(context, 1, &source, sourceSize, &err);
  if(err != 0) {printf("Error: clCreatProgramWithSource %i\n",err);}
  //"-cl-opt-disable"
  printf("\tJIT Compiling: %s\n\t on Device %i\n",prog_name,*device_id);
  err = clBuildProgram(*program, 1, device_id, "-Werror", NULL, NULL);
  if(err != 0) {
    printf("Error: clBuildProgram %i\n",err);
    size_t log_size = 0;
    clGetProgramBuildInfo(*program, *device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    const char log[] = {log_size};
    clGetProgramBuildInfo(*program, *device_id, CL_PROGRAM_BUILD_LOG, log_size, (void*)log, NULL);
    printf("clError Log: %s",log); //dump to screen
  }
  fflush(stdout);
  return err;
}

cl_int clinterface_buildkernels(cl_device_id* devices)
{
    cl_int err;

    // 1D Reduction Routines
    char* program_path2 = "./gpu_kernels/clreduce.cl";
    err = clinterface_clbuild(gContext, devices, &clreduce, program_path2);

    // 1D GPU CGSTAB routines
    char* program_path3 = "./gpu_kernels/clcgstab1D.cl";
    err = clinterface_clbuild(gContext, devices, &clcgstab1D, program_path3);

    //clreduce routines
    clreduce_1d = clCreateKernel(clreduce, "clreduce_1d", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clreduce_abs_1d = clCreateKernel(clreduce, "clreduce_abs_1d", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }

    //clcgstab1D routines
    clcgstab1D_zer = clCreateKernel(clcgstab1D, "clcgstab1D_zer", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_irv = clCreateKernel(clcgstab1D, "clcgstab1D_irv", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_pmd = clCreateKernel(clcgstab1D, "clcgstab1D_pmd", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_pmd2 = clCreateKernel(clcgstab1D, "clcgstab1D_pmd2", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_iwa = clCreateKernel(clcgstab1D, "clcgstab1D_iwa", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_cbo = clCreateKernel(clcgstab1D, "clcgstab1D_cbo", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_cpk = clCreateKernel(clcgstab1D, "clcgstab1D_cpk", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_sfs = clCreateKernel(clcgstab1D, "clcgstab1D_sfs", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_sfs2 = clCreateKernel(clcgstab1D, "clcgstab1D_sfs2", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_sbs = clCreateKernel(clcgstab1D, "clcgstab1D_sbs", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_sbs2 = clCreateKernel(clcgstab1D, "clcgstab1D_sbs2", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_cuk = clCreateKernel(clcgstab1D, "clcgstab1D_cuk", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_ruk = clCreateKernel(clcgstab1D, "clcgstab1D_ruk", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_ufr = clCreateKernel(clcgstab1D, "clcgstab1D_ufr", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_smy = clCreateKernel(clcgstab1D, "clcgstab1D_smy", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_smy2 = clCreateKernel(clcgstab1D, "clcgstab1D_smy2", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_cav = clCreateKernel(clcgstab1D, "clcgstab1D_cav", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_caf = clCreateKernel(clcgstab1D, "clcgstab1D_caf", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_cfr = clCreateKernel(clcgstab1D, "clcgstab1D_cfr", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }
    clcgstab1D_sfs_init = clCreateKernel(clcgstab1D, "clcgstab1D_sfs_init", &err);
    if (err != 0)
    {
      printf("Error: Failed to create compute kernel! %i\n",err);
      return err;
    }

    //clreduce routines
    kernel_list[22] = clreduce_1d;
    kernel_list[23] = clreduce_abs_1d;

    //clreduce routines
    kernel_list[27] = clcgstab1D_zer; 
    kernel_list[28] = clcgstab1D_irv; 
    kernel_list[29] = clcgstab1D_pmd; 
    kernel_list[30] = clcgstab1D_iwa; 
    kernel_list[31] = clcgstab1D_cbo; 
    kernel_list[32] = clcgstab1D_cpk; 
    kernel_list[33] = clcgstab1D_sfs; 
    kernel_list[34] = clcgstab1D_sbs; 
    kernel_list[35] = clcgstab1D_cuk; 
    kernel_list[36] = clcgstab1D_ruk; 
    kernel_list[37] = clcgstab1D_ufr; 
    kernel_list[38] = clcgstab1D_smy; 
    kernel_list[39] = clcgstab1D_cav; 
    kernel_list[40] = clcgstab1D_caf; 
    kernel_list[41] = clcgstab1D_cfr;
    kernel_list[42] = clcgstab1D_sfs2;
    kernel_list[43] = clcgstab1D_smy2;
    kernel_list[44] = clcgstab1D_sbs2;
    kernel_list[45] = clcgstab1D_pmd2;
    kernel_list[46] = clcgstab1D_sfs_init;
    return err;

}

cl_int clinterface_get_devices(cl_device_id* devices,
	cl_int*     size,
	cl_int*    numPtr )
{
    cl_int err;
    size_t n;
    err = clGetContextInfo(gContext, CL_CONTEXT_DEVICES,*size*sizeof(cl_device_id), devices, &n );
    //printf("size,n = %i, %lu\n",*size,n);
    *numPtr = n / sizeof(cl_device_id);
    return err;
}

cl_int clinterface_queue_create( cl_device_id* device, cl_command_queue* queuePtr )
{
    cl_int err;
    if(queuePtr==NULL) return -1;
    *queuePtr = clCreateCommandQueue( gContext, device[0], 0, &err );
    // printf("queue_ptr %p\n",queuePtr);
    return err;
}

cl_int clinterface_queue_create_async( cl_device_id* device, cl_command_queue* queuePtr )
{
    cl_int err;
    if(queuePtr==NULL) return -1;
    *queuePtr = clCreateCommandQueue( gContext, device[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err );
    // printf("queue_ptr %p\n",queuePtr);
    return err;
}

cl_int clinterface_tp_uint_(size_t* size )
{
    cl_int err;
    //fflush(stdout);
    //printf("2: buffer size %zu\n", *size);
    //fflush(stdout);
    return 0;
}

cl_int clinterface_malloc( cl_mem* ptrPtr, size_t* size )
{
    cl_int err;
    //printf("buffer size %lu\n", *size);
    *ptrPtr = clCreateBuffer( gContext, CL_MEM_READ_WRITE, *size, NULL, &err );
    return err;
}

cl_int clinterface_malloc_readonly( cl_mem* ptrPtr, size_t* size )
{
    cl_int err;
    //printf("buffer size %lu\n", *size);
    *ptrPtr = clCreateBuffer( gContext, CL_MEM_READ_ONLY, *size, NULL, &err );
    return err;
}

cl_int clinterface_malloc_writeonly( cl_mem* ptrPtr, size_t* size )
{
    cl_int err;
    //printf("buffer size %lu\n", *size);
    *ptrPtr = clCreateBuffer( gContext, CL_MEM_WRITE_ONLY, *size, NULL, &err );
    return err;
}

//All mapped commands used the HOST defined memory
cl_int clinterface_malloc_mapped( cl_mem* ptrPtr, size_t* size, void* ptrHost )
{
    cl_int err;
    //printf("buffer size %lu\n", *size);
    *ptrPtr = clCreateBuffer( gContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, *size, ptrHost, &err );
    return err;
}

cl_int clinterface_malloc_mapped_readonly( cl_mem* ptrPtr, size_t* size, void* ptrHost )
{
    cl_int err;
    //printf("buffer size %lu\n", *size);
    *ptrPtr = clCreateBuffer( gContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, *size, ptrHost, &err );
    return err;
}

cl_int clinterface_malloc_mapped_writeonly( cl_mem* ptrPtr, size_t* size, void* ptrHost )
{
    cl_int err;
    //printf("buffer size %lu\n", *size);
    *ptrPtr = clCreateBuffer( gContext, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, *size, ptrHost, &err );
    return err;
}

//All pinned commands will make a HOST defined memory spot to use
cl_int clinterface_malloc_pinned( cl_mem* ptrPtr, size_t* size )
{
    cl_int err;
    cl_mem memobj;
    memobj = clCreateBuffer( gContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, *size, NULL, &err );
    ptrPtr = (cl_mem *)memobj;
    return err;
}

cl_int clinterface_malloc_pinned_readonly( cl_mem* ptrPtr, size_t* size )
{
    cl_int err;
    cl_mem memobj;
    memobj = clCreateBuffer( gContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, *size, NULL, &err );
    ptrPtr = (cl_mem *)memobj;
    return err;
}

cl_int clinterface_malloc_pinned_writeonly( cl_mem* ptrPtr, size_t* size )
{
    cl_int err;
    cl_mem memobj;
    memobj = clCreateBuffer( gContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, *size, NULL, &err );
    ptrPtr = (cl_mem *)memobj;
    return err;
}

cl_int clinterface_free( cl_mem ptr )
{
    cl_int err = clReleaseMemObject( ptr );
    return err;
}

cl_int
clinterface_zsetmatrix(
    cl_int *m, cl_int *n,
    cl_double2* hA_src, size_t* hA_offset, cl_int* ldha,
    cl_mem* dA_dst, size_t* dA_offset, cl_int* ldda,
    cl_command_queue* queue )
{
    size_t buffer_origin[3] = { *dA_offset*sizeof(cl_double2), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { *m*sizeof(cl_double2), *n, 1 };
    gNumEvents++; //increment the event log
    cl_int err = clEnqueueWriteBufferRect(
        *queue, *dA_dst, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        *ldda*sizeof(cl_double2), 0,
        *ldha*sizeof(cl_double2), 0,
        hA_src, 0, NULL, &gEvents[gNumEvents] );
    return err;
}

cl_int
clinterface_zsetmatrix_mapped(
    cl_int *m, cl_int *n,
    void* hA_src, size_t* hA_offset, cl_int* ldha,
    cl_mem* dA_dst, size_t* dA_offset, cl_int* ldda,
    cl_command_queue* queue )
{
    cl_int err;
    gNumEvents++; //increment the event log
    hA_src = clEnqueueMapBuffer(
        *queue, *dA_dst, CL_TRUE,  // blocking
        NULL, 0,
        *m**n*sizeof(cl_double2), 0,
        NULL, &gEvents[gNumEvents],
        &err );
    return err;
}
cl_int clinterface_zsetvector(
	cl_int* n, 
	cl_double2* hA_src, size_t* hA_offset, cl_int* incx, 
	cl_mem* dA_dst, size_t* dA_offset, cl_int* incy, 
	cl_command_queue* queue )
{
	cl_int err;
	if(*incx == 1 && *incy == 1){
                gNumEvents++; //increment the event log
                err = clEnqueueWriteBuffer(
				*queue, *dA_dst, CL_TRUE, 
				0, *n*sizeof(cl_double2), 
				hA_src, 0, NULL, &gEvents[gNumEvents]);
		return err;
	}else{
		cl_int ldha = *incx;
		cl_int ldda = *incy;
                cl_int nn = 1;
		err = clinterface_zsetmatrix(n, &nn, 
					hA_src, hA_offset, &ldha, 
					dA_dst, dA_offset, &ldda, 
					queue);
		return err;
	}
}
cl_int
clinterface_zgetmatrix(
    cl_int *m, cl_int *n,
    cl_mem* dA_src, size_t* dA_offset, cl_int* ldda,
    cl_double2* hA_dst, size_t* hA_offset, cl_int* ldha,
    cl_command_queue* queue )
{
    size_t buffer_origin[3] = { *dA_offset*sizeof(cl_double2), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { *m*sizeof(cl_double2), *n, 1 };
    gNumEvents++; //increment the event log
    cl_int err = clEnqueueReadBufferRect(
        *queue, *dA_src, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        *ldda*sizeof(cl_double2), 0,
        *ldha*sizeof(cl_double2), 0,
        hA_dst, 0, NULL, &gEvents[gNumEvents] );
    return err;
}
//
//
// Integer Routines
//
//
cl_int
clinterface_isetmatrix(
    cl_int *m, cl_int *n,
    cl_int* hA_src, size_t* hA_offset, cl_int* ldha,
    cl_mem* dA_dst, size_t* dA_offset, cl_int* ldda,
    cl_command_queue* queue )
{
    size_t buffer_origin[3] = { *dA_offset*sizeof(cl_int), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { *m*sizeof(cl_int), *n, 1 };
    gNumEvents++; //increment the event log
    cl_int err = clEnqueueWriteBufferRect(
        *queue, *dA_dst, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        *ldda*sizeof(cl_int), 0,
        *ldha*sizeof(cl_int), 0,
        hA_src, 0, NULL, &gEvents[gNumEvents] );
    return err;
}
cl_int
clinterface_isetmatrix_nb(
    cl_int *m, cl_int *n,
    cl_int* hA_src, size_t* hA_offset, cl_int* ldha,
    cl_mem* dA_dst, size_t* dA_offset, cl_int* ldda,
    cl_command_queue* queue )
{
    size_t buffer_origin[3] = { *dA_offset*sizeof(cl_int), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { *m*sizeof(cl_int), *n, 1 };
    gNumEvents++; //increment the event log
    cl_int err = clEnqueueWriteBufferRect(
        *queue, *dA_dst, CL_FALSE,  // non-blocking
        buffer_origin, host_orig, region,
        *ldda*sizeof(cl_int), 0,
        *ldha*sizeof(cl_int), 0,
        hA_src, 0, NULL, &gEvents[gNumEvents] );
    return err;
}
cl_int clinterface_isetvector(
	cl_int* n, 
	cl_int* hA_src, size_t* hA_offset, cl_int* incx, 
	cl_mem* dA_dst, size_t* dA_offset, cl_int* incy, 
	cl_command_queue* queue )
{
	cl_int err;
	if(*incx == 1 && *incy == 1){
                gNumEvents++; //increment the event log
                err = clEnqueueWriteBuffer(
				*queue, *dA_dst, CL_TRUE, 
				0, *n*sizeof(cl_int), 
				hA_src, 0, NULL, &gEvents[gNumEvents] );
		return err;
	}else{
		cl_int ldha = *incx;
		cl_int ldda = *incy;
                cl_int nn = 1;
		err = clinterface_isetmatrix(n, &nn, 
					hA_src, hA_offset, &ldha, 
					dA_dst, dA_offset, &ldda, 
					queue);
		return err;
	}
}
cl_int
clinterface_igetmatrix(
    cl_int *m, cl_int *n,
    cl_mem* dA_src, size_t* dA_offset, cl_int* ldda,
    cl_int* hA_dst, size_t* hA_offset, cl_int* ldha,
    cl_command_queue* queue )
{
    size_t buffer_origin[3] = { *dA_offset*sizeof(cl_int), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { *m*sizeof(cl_int), *n, 1 };
    gNumEvents++; //increment the event log
    cl_int err = clEnqueueReadBufferRect(
        *queue, *dA_src, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        *ldda*sizeof(cl_int), 0,
        *ldha*sizeof(cl_int), 0,
        hA_dst, 0, NULL, &gEvents[gNumEvents] );
    return err;
}
cl_int
clinterface_igetmatrix_nb(
    cl_int *m, cl_int *n,
    cl_mem* dA_src, size_t* dA_offset, cl_int* ldda,
    cl_int* hA_dst, size_t* hA_offset, cl_int* ldha,
    cl_command_queue* queue )
{
    size_t buffer_origin[3] = { *dA_offset*sizeof(cl_int), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { *m*sizeof(cl_int), *n, 1 };
    gNumEvents++; //increment the event log
    cl_int err = clEnqueueReadBufferRect(
        *queue, *dA_src, CL_FALSE,  // non-blocking
        buffer_origin, host_orig, region,
        *ldda*sizeof(cl_int), 0,
        *ldha*sizeof(cl_int), 0,
        hA_dst, 0, NULL, &gEvents[gNumEvents] );
    return err;
}
cl_int 
clinterface_igetvector(
	cl_int *n, 
	cl_mem* dA_src, size_t* dA_offset, cl_int* incx, 
	cl_int* hA_dst, size_t* hA_offset, cl_int* incy,
	cl_command_queue* queue )
{
	cl_int err;
	if(*incx == 1 && *incy == 1){
                gNumEvents++; //increment the event log
		err = clEnqueueReadBuffer(
					*queue, *dA_src, CL_TRUE, 
					*dA_offset*sizeof(cl_int), *n*sizeof(cl_int), 
					hA_dst, 0, NULL, &gEvents[gNumEvents] );
		return err;			
	}else{
		cl_int ldda = *incx;
		cl_int ldha = *incy;
                cl_int nn = 1;
		err = clinterface_igetmatrix(n, &nn, 
						dA_src, dA_offset, &ldda,
						hA_dst, hA_offset, &ldha,
						queue);
		return err;
	}
}
//
//
// Double Routines
//
//
cl_int
clinterface_dsetmatrix(
    cl_int *m, cl_int *n,
    cl_double* hA_src, size_t* hA_offset, cl_int* ldha,
    cl_mem* dA_dst, size_t* dA_offset, cl_int* ldda,
    cl_command_queue* queue )
{
    size_t buffer_origin[3] = { *dA_offset*sizeof(cl_double), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { *m*sizeof(cl_double), *n, 1 };
    gNumEvents++; //increment the event log
    cl_int err = clEnqueueWriteBufferRect(
        *queue, *dA_dst, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        *ldda*sizeof(cl_double), 0,
        *ldha*sizeof(cl_double), 0,
        hA_src, 0, NULL, &gEvents[gNumEvents] );
    return err;
}
cl_int
clinterface_dsetmatrix_nb(
    cl_int *m, cl_int *n,
    cl_double* hA_src, size_t* hA_offset, cl_int* ldha,
    cl_mem* dA_dst, size_t* dA_offset, cl_int* ldda,
    cl_command_queue* queue )
{
    size_t buffer_origin[3] = { *dA_offset*sizeof(cl_double), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { *m*sizeof(cl_double), *n, 1 };
    gNumEvents++; //increment the event log
    cl_int err = clEnqueueWriteBufferRect(
        *queue, *dA_dst, CL_FALSE,  // non-blocking
        buffer_origin, host_orig, region,
        *ldda*sizeof(cl_double), 0,
        *ldha*sizeof(cl_double), 0,
        hA_src, 0, NULL, &gEvents[gNumEvents] );
    return err;
}
cl_int clinterface_dsetvector(
	cl_int* n, 
	cl_double* hA_src, size_t* hA_offset, cl_int* incx, 
	cl_mem* dA_dst, size_t* dA_offset, cl_int* incy, 
	cl_command_queue* queue )
{
	cl_int err;
	if(*incx == 1 && *incy == 1){
                gNumEvents++; //increment the event log
                err = clEnqueueWriteBuffer(
				*queue, *dA_dst, CL_TRUE, 
				0, *n*sizeof(cl_double), 
				hA_src, 0, NULL, &gEvents[gNumEvents] );
		return err;
	}else{
		cl_int ldha = *incx;
		cl_int ldda = *incy;
                cl_int nn = 1;
		err = clinterface_dsetmatrix(n, &nn, 
					hA_src, hA_offset, &ldha, 
					dA_dst, dA_offset, &ldda, 
					queue);
		return err;
	}
}
cl_int
clinterface_dgetmatrix(
    cl_int *m, cl_int *n,
    cl_mem* dA_src, size_t* dA_offset, cl_int* ldda,
    cl_double* hA_dst, size_t* hA_offset, cl_int* ldha,
    cl_command_queue* queue )
{
    size_t buffer_origin[3] = { *dA_offset*sizeof(cl_double), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { *m*sizeof(cl_double), *n, 1 };
    gNumEvents++; //increment the event log
    cl_int err = clEnqueueReadBufferRect(
        *queue, *dA_src, CL_TRUE,  // blocking
        buffer_origin, host_orig, region,
        *ldda*sizeof(cl_double), 0,
        *ldha*sizeof(cl_double), 0,
        hA_dst, 0, NULL, &gEvents[gNumEvents] );
    return err;
}

cl_int
clinterface_dgetmatrix_nb(
    cl_int *m, cl_int *n,
    cl_mem* dA_src, size_t* dA_offset, cl_int* ldda,
    cl_double* hA_dst, size_t* hA_offset, cl_int* ldha,
    cl_command_queue* queue )
{
    size_t buffer_origin[3] = { *dA_offset*sizeof(cl_double), 0, 0 };
    size_t host_orig[3]     = { 0, 0, 0 };
    size_t region[3]        = { *m*sizeof(cl_double), *n, 1 };
    gNumEvents++; //increment the event log
    cl_int err = clEnqueueReadBufferRect(
        *queue, *dA_src, CL_FALSE,  // non-blocking
        buffer_origin, host_orig, region,
        *ldda*sizeof(cl_double), 0,
        *ldha*sizeof(cl_double), 0,
        hA_dst, 0, NULL, &gEvents[gNumEvents] );
    return err;
}

cl_int 
clinterface_dgetvector(
	cl_int *n, 
	cl_mem* dA_src, size_t* dA_offset, cl_int* incx, 
	cl_double* hA_dst, size_t* hA_offset, cl_int* incy,
	cl_command_queue* queue )
{
	cl_int err;
        gNumEvents++; //increment the event log
	err = clEnqueueReadBuffer(*queue, *dA_src, CL_TRUE, 
				*dA_offset*sizeof(cl_double), *n*sizeof(cl_double), 
				hA_dst, 0, NULL, &gEvents[gNumEvents] );
	return err;			
}
//
//
// General Routines
//
//
//
cl_int clinterface_clflush(cl_command_queue *queue)
{
   cl_int err;
   err = clFlush(*queue);
   return err;
}

cl_int clinterface_clfinish(cl_command_queue *queue)
{
   cl_int err;
   err = clFinish(*queue);
   return err;
}

cl_int clinterface_clbarrier(cl_command_queue *queue)
{
   cl_int err;
   //printf("number of events in queue = %i\n",gNumEvents);
   //err =  clEnqueueBarrierWithWaitList(*queue,gNumEvents,&gEvents[1],NULL);
   //err = clWaitForEvents(1, &events[0]);
   //err = clReleaseEvent(events[0]);
   err=clFinish(*queue);
   gNumEvents = 0; //reset number of events
   return err;
}

cl_int clinterface_clretainmemobject(cl_mem* dA_src)
{
   cl_int err;
   err = clRetainMemObject(*dA_src);
   return err;
}
cl_int clinterface_clreleasememobject(cl_mem* dA_src)
{
   cl_int err;
   err = clReleaseMemObject(*dA_src);
   return err;
}

cl_int clinterface_clsetkernelarg(int *knum, cl_uint *carg, size_t *pt, void *array)
{
  cl_int err;
  //printf("arg num,size,: %hi %hi %hi\n",*carg,*pt,sizeof(cl_mem));
  //                    kernel,      command_num, size, array_ptr
  err = clSetKernelArg(kernel_list[*knum], *carg, *pt, (void *)&(array));
  return err;
}

cl_int clinterface_clsetkernelarg_ptr(int *knum, cl_uint *carg, size_t *pt, void *array)
{
  cl_int err;
  //printf("arg num,size,: %hi %hi %hi\n",*carg,*pt,sizeof(cl_mem));
  //                    kernel,      command_num, size, array_ptr
  err = clSetKernelArg(kernel_list[*knum], *carg, *pt, array);
  return err;
}

cl_int clinterface_clsetkernelarg_shmem(int *knum, cl_uint *carg, size_t *pt)
{
  cl_int err;
  //printf("arg num,size,: %hi %hi %hi\n",*carg,*pt,sizeof(cl_mem));
  //                    kernel,      command_num, size, array_ptr
  err = clSetKernelArg(kernel_list[*knum], *carg, *pt, NULL);
  return err;
}

cl_int clinterface_clenqueuendrangekernel(int *knum, int *dim, size_t *globalThreads, size_t *localThreads, cl_command_queue *queue)
{
  cl_int err;
  //printf("globalThreads,localThreads,: %hi %hi %hi\n",globalThreads[0],localThreads[0],*dim);
  //printf("globalThreads,localThreads,: %hi %hi\n",globalThreads[1],localThreads[1]);
  //printf("globalThreads,localThreads,: %hi %hi\n",globalThreads[2],localThreads[2]);
  //printf("inside clinterface_clenqueuendrangekernel\n");
  //printf("knum,kernel: %i %i\n",*knum,kernel_list[*knum]);
  //fflush(0);
  //                                                  dim, offset,global_work_size,local_work_size,
  if(gNumEvents>0) {
  //  printf("number of events in queue = %i\n",gNumEvents);
    err = clEnqueueNDRangeKernel(*queue, kernel_list[*knum], *dim, NULL, globalThreads, localThreads, gNumEvents, &gEvents[1], &events[0]);
  }else{
    err = clEnqueueNDRangeKernel(*queue, kernel_list[*knum], *dim, NULL, globalThreads, localThreads, 0, NULL, &events[0]);
  }
  err = clWaitForEvents(1, &events[0]);
  //err = clReleaseEvent(events[0]);
  //err=clFinish(*queue);
  gNumEvents = 0; //reset number of events
  return err;
}

}
