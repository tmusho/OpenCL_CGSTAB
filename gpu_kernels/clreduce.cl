// OpenCL Version of a Conguent Gradient Iterative Solver
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
//#pragma OPENCL EXTENSION cl_khr_fp64: enable
//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

//CALCULATE VECTOR REDUCTION
__kernel void clreduce_1d(__constant int* nxyz, __global int* corder, __global double* RESO, __global double* AE, __local double *shmem)
{
  int idx,bidx,bszx,i,ijk;
  int offset;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    shmem[idx] = AE[ijk];
  }else {
    shmem[idx] = 0.0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for(offset = bszx/2; offset > 0; offset = offset/2) {
    if (idx < offset) {
      shmem[idx] += shmem[idx + offset];
      shmem[idx + offset] = 0.0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (idx == 0 ){
    RESO[bidx] = shmem[0];
  }


}

//CALCULATE VECTOR REDUCTION
__kernel void clreduce_abs_1d(__constant int* nxyz, __global int* corder, __global double* RESO, __global double* AE, __local double *shmem)
{
  int idx,bidx,bszx,i,ijk;
  int offset;

  idx = get_local_id(0); //thread id - i-dir
  bidx = get_group_id(0); //group id - i-dir
  bszx = get_local_size(0); //group size - i-dir

  i = bidx*bszx+idx;
  ijk = corder[i]; //must lie on grid

  if(ijk<*nxyz) {
    shmem[idx] = fabs(AE[ijk]);
  }else {
    shmem[idx] = 0.0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for(offset = bszx/2; offset > 0; offset = offset/2) {
    if (idx < offset) {
      shmem[idx] += shmem[idx + offset];
      shmem[idx + offset] = 0.0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (idx == 0 ){
    RESO[bidx] = shmem[0];
  }


}
