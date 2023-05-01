// OpenCL Version of a Conguent Gradient Iterative Solver
// Written by Terence Musho - West Virginia University
//
//
//  This code was written to work with the DREAM CFD Code
//  The following code are the routines to a Conjugent Gradient Solver
//  The way that this was written was to use the threads in 1 dimensions
//  See gpu_clinterface.f90 for number of threads. This version is simplified
//  over the other clcgstab.cl version.
//
//  Parts of the CGSTAB routine are not parallizable and are still
//  evaluated on the CPU. The GPU speed up the reduction routines
//  and simple scalar time vector routines. At current time the
//  gpus see a ?times increase over cpu.
//
//
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
//

//#pragma OPENCL EXTENSION cl_khr_fp64: enable
//#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_select_fprounding_mode : enable
//#pragma OPENCL SELECT_ROUNDING_MODE rtz

// Double
//typedef double real_t;

//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

/* Use one of these three definitions to perform the multiply-add operation */
#define mmad(x,y,z) (x*y+z)       // Second most exact (or tied with fma), but fastest due to use of FMAC (FMA-accumulator) instruction
//#define mmad(x,y,z) mad(x,y,z)  // Undefined precision (for some cases this can be very very wrong)
//#define mmad(x,y,z) fma(x,y,z)  // Guaranteed to be the most exact

#define mmz(x,y) (x*y)
#define dmad(x,y) (x/y)       //

//ZERO
__kernel void clcgstab1D_zer(__constant int* nxyz, __global double* PK, __global double* ZK, __global double* ZKT, __global double* VK, __global double* UK, 
                            __global double* RES, __global double* RESO, __global double* BET, __global double* BETO, __global double* VV, 
                            __global double* VRES, __global double* D, __global int* corder)
{

  int idx,bidx,bszx;
  int i,ijk;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    PK[ijk]   = 0;
    ZK[ijk]   = 0;
    ZKT[ijk]  = 0;
    VK[ijk]   = 0;
    UK[ijk]   = 0;

    RES[ijk]  = 0;
    RESO[ijk] = 0;
    BET[ijk]  = 0;
    BETO[ijk] = 0;
    VV[ijk]   = 0;
    VRES[ijk] = 0;
    D[ijk] = 0;
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
  }
}

//CALCULATE INITIAL RESIDUAL VECTOR
__kernel void clcgstab1D_irv(__constant int* nxyz, __constant int* NIJ, __constant int* NJ, __global double* RES, __global double* Q, __global double* AP, 
                                    __global double* AE, __global double* AW, __global double* AN, __global double* AS, __global double* AT, 
                                    __global double* AB, __global double* FI, __global int* corder)
{

  int idx,bidx,bszx;
  int i;
  int ijk;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    //            RES(IJK)= Q(IJK) - AP(IJK)*FI(IJK)-&
    //     &           AE(IJK)*FI(IJK+NJ) - AW(IJK)*FI(IJK-NJ) -&
    //     &           AN(IJK)*FI(IJK+1)  - AS(IJK)*FI(IJK-1)  -&
    //     &           AT(IJK)*FI(IJK+NIJ)- AB(IJK)*FI(IJK-NIJ)
    //P_1 = (AP[ijk]*FI[ijk]);
    //P_2 = (AE[ijk]*FI[ijk+NJ])+(AW[ijk]*FI[ijk-NJ]);
    //P_3 = (AN[ijk]*FI[ijk+1])+(AS[ijk]*FI[ijk-1]);
    //P_4 = (AT[ijk]*FI[ijk+NIJ])+(AB[ijk]*FI[ijk-NIJ]);
    RES[ijk] = Q[ijk] - mmz(AP[ijk],FI[ijk]) - mmz(AE[ijk],FI[ijk+*NJ]) - mmz(AW[ijk],FI[ijk-*NJ]) 
                      - mmz(AN[ijk],FI[ijk+1]) - mmz(AS[ijk],FI[ijk-1]) - mmz(AT[ijk],FI[ijk+*NIJ])
                      - mmz(AB[ijk],FI[ijk-*NIJ]);
    //RES[ijk]= Q[ijk] - AP[ijk]*FI[ijk]- AE[ijk]*FI[ijk+NJ] - AW[ijk]*FI[ijk-NJ] - AN[ijk]*FI[ijk+1]  - AS[ijk]*FI[ijk-1]  -  AT[ijk]*FI[ijk+NIJ]- AB[ijk]*FI[ijk-NIJ];
    //write_mem_fence(CLK_GLOBAL_MEM_FENCE);
  }

}

//CALCULATE ELEMENTS OF PRECONDITIONING MATRIX DIAGONAL
__kernel void clcgstab1D_pmd(__constant int* nxyz, __constant int* nxmax, __constant int* nymax, __constant int* nzmax, int NIJ, int NJ, __global int* LK, __global int* LI, __global double* D, __global double* AP, 
                                    __global double* AE, __global double* AW, __global double* AN, __global double* AS, __global double* AT, 
                                    __global double* AB, __global double* D_T, __global int* corder)
{

  int idx,bidx,bszx;
  int i;
  int ijk;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    //D_T[ijk] = 1./(AP[ijk]-AW[ijk]*D[ijk-NJ]*AE[ijk-NJ]-AS[ijk]*D[ijk-1]*AN[ijk-1]-AB[ijk]*D[ijk-NIJ]*AT[ijk-NIJ]);
  }
}

//CALCULATE ELEMENTS OF PRECONDITIONING MATRIX DIAGONAL PART2
__kernel void clcgstab1D_pmd2( __constant int* nxyz, __constant int* nxmax, __constant int* nymax, __constant int* nzmax, __constant int* NIJ,
                              __constant int* NJ, __global int* LK, __global int* LI, __global double* D, __global double* D_T, __global int* corder)
{

  int idx,bidx,bszx;
  int i;
  int ijk;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    //D[ijk] = D_T[ijk];
  }

}

//INITIALIZE WORKING ARRAYS AND CONSTANTS
__kernel void clcgstab1D_iwa(__constant int* nxyz, __global double* PK, __global double* UK, 
                             __global double* ZK, __global double* VK, __global double* RES, __global double* RESO, __global int* corder)
{

  int idx,bidx,bszx;
  int i;
  int ijk;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    RESO[ijk]=RES[ijk];
    PK[ijk]=0;
    UK[ijk]=0;
    ZK[ijk]=0;
    VK[ijk]=0;
    //write_mem_fence(CLK_GLOBAL_MEM_FENCE);
  }

}

//CALCULATE BETA AND OMEGA
__kernel void clcgstab1D_cbo(__constant int* nxyz, __global double* BET, __global double* RES, __global double* RESO, __global int* corder)
{

  int idx,bidx,bszx;
  int i;
  int ijk;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    BET[ijk] = RES[ijk]*RESO[ijk];
  }

}

//CALCULATE PK
__kernel void clcgstab1D_cpk(__constant int* nxyz, __global double* PK, __global double* RES, 
                             __global double* UK, __global int* corder, __constant  double* OM, __constant  double* ALF)
{

  int idx,bidx,bszx;
  int i;
  int ijk;
  double PK_T, UK_T;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    PK_T=PK[ijk];
    UK_T=UK[ijk];
    PK[ijk] = RES[ijk]+(*OM*(PK_T-*ALF*UK_T));
  }

}
//Initialize FORWARD SUB
__kernel void clcgstab1D_sfs_init(__global int* block_finish){

  int idx,bidx;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir

  if(idx==0) block_finish[bidx]=0; //not done
  write_mem_fence(CLK_GLOBAL_MEM_FENCE);

}
//SOLVE (M ZK = PK) - FORWARD SUBSTITUTION
__kernel void clcgstab1D_sfs(__constant int* nxyz, __constant int* nymax, __constant int* NIJ, __constant int* NJ, __global double* PK, __global double* AW, __global double* ZK, __global double* AS,
                                   __global double* AB, __global double* D, __global int* block_finish, __global int* forder, __local double *shmem_s, __local double *shmem_w, 
                                   __local double *shmem_b, __local double *shmem, __local int *waiting)
{

  int idx,bidx,bszx,bido;
  int i, j;
  int curr_row;
  int ngrp_col;
  int rmthd_col;
  int ijk;

  __local int lock; //local lock shared between threads in block

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  //This calculation will take chunks of 16 from the matrix. Each thread will get 
  // a part of the 16. The blocks to the left and above must be completed before 
  // the next block can proceed.
  bido=bidx;
  bidx=forder[bido];

  ngrp_col = ((*nymax-2)+(bszx-1))/bszx; //absolute number of local-blocks that correspond to a row of the grid
  curr_row = bidx/ngrp_col;
  rmthd_col = ngrp_col*bszx-(*nymax-2); //remaining threads in last block of row

  //each thread will get a row  
  i = bidx*bszx+idx-curr_row*rmthd_col;
  //index of main diagonal
  ijk = i+*NIJ+*NJ+1; //must lie on grid

  if((bidx+1)%ngrp_col==0){
    bszx=bszx-rmthd_col;
    if(idx>=bszx) ijk=*nxyz;
  }
  //initial local locks
  if(idx==0) {
    waiting[0]=1; //go
  }else{
    waiting[idx]=0; //wait
  }

  //zero shared memory
  shmem_s[idx]=0;
  shmem_w[idx]=0;
  shmem_b[idx]=0;
  shmem[idx]=0;
  barrier(CLK_LOCAL_MEM_FENCE);

  //this loop will spin-lock the groups
  //it will increment the group id until it finds an eligible block
  //once a group is eligible it will save 2D neighbors to shared memory
  j=1;
  while(j>0) { //timer will timeout after ngrp loops
    //blocks above and to left must be finished
    //this makes the whole grid look like 2D. We 
    //could speed this up by taking into account the 
    //third dimension.
    if(idx==0){
      if(bido==0){ 
        lock=0;
      }else if(atomic_min(&block_finish[bido-1],1)<0){//BLOCKS IN FIRST ROW
        lock=0;
      }else{
        lock=1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE); //sync lock

    if(lock==0) { //first row
      read_mem_fence(CLK_GLOBAL_MEM_FENCE);
      shmem_s[idx] = (AS[ijk]*ZK[ijk-1]);   //south
      shmem_w[idx] = (AW[ijk]*ZK[ijk-*NJ]);  //west
      shmem_b[idx] = (AB[ijk]*ZK[ijk-*NIJ]); //bottom
      break;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

    if(idx==0){
      shmem[idx] = ((PK[ijk]-shmem_w[idx]-shmem_s[idx]-shmem_b[idx])*D[ijk]);
      waiting[idx+1]=1; //start
      waiting[idx]=0; //done
    }

    //this shouldn't take more then bszx iterations
    for(j=1;j<bszx;j++) {
      barrier(CLK_LOCAL_MEM_FENCE);
      //thread must wait for thread to left to finish
      if(waiting[idx]==1){
        shmem[idx] = ((PK[ijk]-shmem_w[idx]-AS[ijk]*shmem[idx-1]-shmem_b[idx])*D[ijk]);
        if(j<bszx-1) waiting[idx+1]=1; //start
        waiting[idx]=0; //done
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if(idx==0){
      for(j=0;j<bszx;j++) {
        i = (bidx*get_local_size(0)+j)-curr_row*rmthd_col;
        ijk = i+*NIJ+*NJ+1; //must lie on grid
        ZK[ijk]=shmem[j];
      }
      write_mem_fence(CLK_GLOBAL_MEM_FENCE);
      atomic_xchg(&block_finish[bido],-1); //block finished
    }

}

__kernel void clcgstab1D_sfs2(__constant int* nxyz, __global double* ZK, __global double* D, __global int* corder)
{

  int idx,bidx,bszx;
  int i;
  int ijk;
  double SMALL;
  double D_T, ZKT;

  SMALL = 1.0E-30;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;

  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    D_T = D[ijk] + SMALL;
    //ZK[ijk] = dmad(ZK[ijk],D_T);
    ZKT = ZK[ijk];
    ZK[ijk] = ZKT/D_T;
  }

}

//SOLVE BACKWARD SUBSTITUTION
__kernel void clcgstab1D_sbs(__constant int* nxyz, __constant int* nymax, __constant int* NIJ, __constant int* NJ, __global double* PK, __global double* AE, __global double* ZK, __global double* AN,
                                   __global double* AT, __global double* D, __global int* block_finish, __global int* forder, __local double *shmem_n, __local double *shmem_e, 
                                   __local double *shmem_t, __local double *shmem, __local int *waiting)
{

  int idx,bidx,bszx,bido;
  int i, j;
  int curr_row;
  int ngrp,ngrp_col;
  int rmthd_col;
  int ijk;

  __local int lock; //local lock shared between threads in block

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir
  ngrp = get_num_groups(0); //number of groups

  //This calculation will take chunks of 16 from the matrix. Each thread will get 
  // a part of the 16. The blocks to the left and above must be completed before 
  // the next block can proceed.

  bido=bidx;
  bidx=forder[ngrp-bido-1];

  ngrp_col = ((*nymax-2)+(bszx-1))/bszx; //absolute number of local-blocks that correspond to a row of the grid
  curr_row = bidx/ngrp_col;
  rmthd_col = ngrp_col*bszx-(*nymax-2); //remaining threads in last block of row

  //each thread will get a row  
  i = bidx*bszx+idx-curr_row*rmthd_col;
  //index of main diagonal
  ijk = i+*NIJ+*NJ+1; //must lie on grid

  if((bidx+1)%ngrp_col==0){
    bszx=bszx-rmthd_col;
    if(bszx-idx<0) ijk=*nxyz;
  }
  //initial local locks
  waiting[idx]=0; //wait

  //zero shared memory
  shmem_n[idx]=0;
  shmem_e[idx]=0;
  shmem_t[idx]=0;
  shmem[idx]=0;
  barrier(CLK_LOCAL_MEM_FENCE);

  //this loop will spin-lock the groups
  //it will increment the group id until it finds an eligible block
  //once a group is eligible it will save 2D neighbors to shared memory
  j=1;
  while(j>0) { //timer will timeout after ngrp loops
    //blocks above and to left must be finished
    //this makes the whole grid look like 2D. We 
    //could speed this up by taking into account the 
    //third dimension.
    if(idx==0){
      if(bido==0){ 
        lock=0;
      }else if(atomic_min(&block_finish[bido-1],1)<0){//BLOCKS IN FIRST ROW
        lock=0;
      }else{
        lock=1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE); //sync lock

    if(lock==0) { //first row
      read_mem_fence(CLK_GLOBAL_MEM_FENCE);
      shmem_n[idx] = (AN[ijk]*ZK[ijk+1]);   //south
      shmem_e[idx] = (AE[ijk]*ZK[ijk+*NJ]);  //west
      shmem_t[idx] = (AT[ijk]*ZK[ijk+*NIJ]); //bottom
      break;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if(idx==bszx-1){
    shmem[idx] = ((ZK[ijk]-shmem_e[idx]-shmem_n[idx]-shmem_t[idx])*D[ijk]);
    waiting[idx-1]=1; //start
    waiting[idx]=0; //done
  }

  //this shouldn't take more then bszx iterations
  for(j=1;j<bszx;j++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    //thread must wait for thread to left to finish
    if(waiting[idx]==1){
      shmem[idx] = ((ZK[ijk]-shmem_e[idx]-AN[ijk]*shmem[idx+1]-shmem_t[idx])*D[ijk]);
      if(j<bszx-1) waiting[idx-1]=1; //start
      waiting[idx]=0; //done
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if(idx==0){
    for(j=0;j<bszx;j++) {
      i = (bidx*get_local_size(0)+j)-curr_row*rmthd_col;
      ijk = i+*NIJ+*NJ+1; //must lie on grid
      ZK[ijk]=shmem[j];
    }
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    atomic_xchg(&block_finish[bido],-1); //block finished
  }

}
__kernel void clcgstab1D_sbs2(__constant int* nxyz, __global double* ZK, __global double* ZK_T, __global int* corder)
{

  int idx,bidx,bszx;
  int i;
  int ijk;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;

  //this routine might not work in parallel on gpus
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    //ZK[ijk]=ZK[ijk];
  }

}

//CALCULATE UK = A.PK
__kernel void clcgstab1D_cuk(__constant int* nxyz, __constant int* NIJ, __constant int* NJ, __global double* UK, __global double* AP, __global double* ZK, __global double* AE,
                                   __global double* AW, __global double* AN, __global double* AS, __global double* AT, __global double* AB, __global int* corder)
{

  int idx,bidx,bszx;
  int i;
  int ijk;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;

  //this routine might not work in parallel on gpus
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    UK[ijk] = (AP[ijk]*ZK[ijk])+(AE[ijk]*ZK[ijk+*NJ])+
              (AW[ijk]*ZK[ijk-*NJ])+(AN[ijk]*ZK[ijk+1])+
              (AS[ijk]*ZK[ijk-1])+(AT[ijk]*ZK[ijk+*NIJ])+
              (AB[ijk]*ZK[ijk-*NIJ]);
  }
}

//CALCULATE SCALAR PRODUCT UK.RESO AND GAMMA
__kernel void clcgstab1D_ruk(__constant int* nxyz, __global double* VV, __global double* RESO, __global double* UK, __global int* corder)
{

  int idx,bidx,bszx;
  int i;
  int ijk;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;

  //this routine might not work in parallel on gpus
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    VV[ijk] = UK[ijk]*RESO[ijk];
  }
}

//UPDATE (FI) AND CALCULATE W (OVERwrite RES - IT IS RES-UPDATE)
__kernel void clcgstab1D_ufr(__constant int* nxyz, __global double* ZK, __global double* RES, __global double* FI, 
                             __global double* UK, __global int* corder,__constant double* GAM)
{

  int idx,bidx,bszx;
  int i;
  int ijk;
  double FI_T, RES_T;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;

  //this routine might not work in parallel on gpus
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    FI_T=FI[ijk];
    FI[ijk] = FI_T+*GAM*ZK[ijk];
    RES_T=RES[ijk];
    RES[ijk] = RES_T-*GAM*UK[ijk];
  }
}

//SOLVE (M Y = W); Y OVERwriteS ZK; FORWARD SUBSTITUTION
__kernel void clcgstab1D_smy(__constant int* nxyz, __constant int* nymax, __constant int* NIJ, __constant int* NJ, __global double* RES, __global double* ZK, __global double* AW, __global double* AS,
                                   __global double* AB, __global double* D, __global int* block_finish, __global int* forder, __local double *shmem_s, __local double *shmem_w, __local double *shmem_b, 
                                   __local double *shmem, __local int *waiting)
{

  int idx,bidx,bszx,bido;
  int i, j;
  int curr_row;
  int ngrp_col;
  int rmthd_col;
  int ijk;

  __local int lock; //local lock shared between threads in block

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  //This calculation will take chunks of 16 from the matrix. Each thread will get 
  // a part of the 16. The blocks to the left and above must be completed before 
  // the next block can proceed.

  bido=bidx;
  bidx=forder[bido];

  ngrp_col = ((*nymax-2)+(bszx-1))/bszx; //absolute number of local-blocks that correspond to a row of the grid
  curr_row = bidx/ngrp_col;
  rmthd_col = ngrp_col*bszx-(*nymax-2); //remaining threads in last block of row

  //each thread will get a row  
  i = bidx*bszx+idx-curr_row*rmthd_col;
  //index of main diagonal
  ijk = i+*NIJ+*NJ+1; //must lie on grid

  if((bidx+1)%ngrp_col==0){
    bszx=bszx-rmthd_col;
    if(idx>=bszx) ijk=*nxyz;
  }
  //initial local locks
  if(idx==0) {
    waiting[0]=1; //go
  }else{
    waiting[idx]=0; //wait
  }
  //zero shared memory
  shmem_s[idx]=0;
  shmem_w[idx]=0;
  shmem_b[idx]=0;
  shmem[idx]=0;
  barrier(CLK_LOCAL_MEM_FENCE);

  //this loop will spin-lock the groups
  //it will increment the group id until it finds an eligible block
  //once a group is eligible it will save 2D neighbors to shared memory
  j=1;
  while(j>0) { //timer will timeout after ngrp loops
    //blocks above and to left must be finished
    //this makes the whole grid look like 2D. We 
    //could speed this up by taking into account the 
    //third dimension.
    if(idx==0){
      if(bido==0){ 
        lock=0;
      }else if(atomic_min(&block_finish[bido-1],1)<0){//BLOCKS IN FIRST ROW
        lock=0;
      }else{
        lock=1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE); //sync lock

    if(lock==0) { //first row
      read_mem_fence(CLK_GLOBAL_MEM_FENCE);
//            ZK(IJK)=(RES(IJK)-AW(IJK)*ZK(IJK-*NJ)-&
//     &              AS(IJK)*ZK(IJK-1)-AB(IJK)*ZK(IJK-*NIJ))*D(IJK)
      shmem_w[idx] = (AW[ijk]*ZK[ijk-*NJ]);  //west
      shmem_s[idx] = (AS[ijk]*ZK[ijk-1]);   //south
      shmem_b[idx] = (AB[ijk]*ZK[ijk-*NIJ]); //bottom
      break;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if(idx==0){
    shmem[idx] = (RES[ijk]-shmem_w[idx]-shmem_s[idx]-shmem_b[idx])*D[ijk];
    waiting[idx+1]=1; //start
    waiting[idx]=0; //done
  }

  //this shouldn't take more then bszx iterations
  for(j=1;j<bszx;j++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    //thread must wait for thread to left to finish
    if(waiting[idx]==1){
      shmem[idx] = (RES[ijk]-shmem_w[idx]-AS[ijk]*shmem[idx-1]-shmem_b[idx])*D[ijk];
      if(j<bszx-1) waiting[idx+1]=1; //start
      waiting[idx]=0; //done
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if(idx==0){
    for(j=0;j<bszx;j++) {
      i = (bidx*get_local_size(0)+j)-curr_row*rmthd_col;
      ijk = i+*NIJ+*NJ+1; //must lie on grid
      ZK[ijk]=shmem[j];
    }
    write_mem_fence(CLK_GLOBAL_MEM_FENCE);
    atomic_xchg(&block_finish[bido],-1); //block finished
  }

}

__kernel void clcgstab1D_smy2(__constant int* nxyz, __global double* ZK, __global double* D, __global int* corder)
{

  int idx,bidx,bszx;
  int i;
  int ijk;
  double SMALL;
  double D_T, ZKT;

  SMALL = 1.E-30;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;

  //this routine might not work in parallel on gpus
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    D_T = D[ijk]+SMALL;
    //ZK[ijk] = dmad(ZK[ijk],D_T);
    ZKT=ZK[ijk];
    ZK[ijk] = ZKT/D_T;
  }

}

//CALCULATE V = A.Y (VK = A.ZK)
__kernel void clcgstab1D_cav(__constant int* nxyz, __constant int* NIJ, __constant int* NJ, __global double* VK, __global double* AP, __global double* ZK, __global double* AE,
                                   __global double* AW, __global double* AN, __global double* AS, __global double* AT, __global double* AB, __global int* corder)
{

  int idx,bidx,bszx;
  int i;
  int ijk;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;

  //this routine might not work in parallel on gpus
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    VK[ijk] = (AP[ijk]*ZK[ijk])+(AE[ijk]*ZK[ijk+*NJ])+
              (AW[ijk]*ZK[ijk-*NJ])+(AN[ijk]*ZK[ijk+1])+
              (AS[ijk]*ZK[ijk-1])+(AT[ijk]*ZK[ijk+*NIJ])+
              (AB[ijk]*ZK[ijk-*NIJ]);
  }
}

//CALCULATE ALPHA (ALF)
__kernel void clcgstab1D_caf(__constant int* nxyz, __global double* VRES, __global double* VK, 
                             __global double* RES, __global double* VV, __global int* corder)
{

  int idx,bidx,bszx;
  int i;
  int ijk;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;

  //this routine might not work in parallel on gpus
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    VRES[ijk] = VK[ijk]*RES[ijk];
    VV[ijk]   = VK[ijk]*VK[ijk];

  }

}

//UPDATE VARIABLE (FI) AND RESIDUAL (RES) VECTORS
__kernel void clcgstab1D_cfr(__constant int* nxyz, __global double* FI, __global double* ZK, 
                             __global double* RES, __global double* RESO, __global double* VK, __global int* corder, __constant double* ALF)
{

  int idx,bidx,bszx;
  int i;
  int ijk;
  double FI_T, RES_T;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;

  //this routine might not work in parallel on gpus
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    FI_T=FI[ijk];
    FI[ijk] = FI_T + *ALF*ZK[ijk];
    RES_T=RES[ijk];
    RES[ijk] = RES_T - *ALF*VK[ijk];
  }

}

