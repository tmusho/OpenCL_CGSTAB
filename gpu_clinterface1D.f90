!  gpu_clinterface.f90 
!
!  Authors: Terence Musho - West Virginia University
!           
!
!  Created: 1/2/16
!  Last Modified: 1/2/16
! 
!
!******************************************************************************
!
!  PROGRAM: Example Program on How to Interface with OpenCL CGSTAB Routines
!
!  PURPOSE: 
! 
!  SUBROUTINES: 
!
!               
!******************************************************************************
!
!Copyright (c) 2016 Dr. Terence Musho - West Virginia University
!
!This program is free software: you can redistribute it and/or modify
!it under the terms of the GNU General Public License as published by
!the Free Software Foundation, either version 3 of the License, or
!(at your option) any later version.
!
!This program is distributed in the hope that it will be useful,
!but WITHOUT ANY WARRANTY; without even the implied warranty of
!MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!GNU General Public License for more details.
!
!You should have received a copy of the GNU General Public License
!along with this program.  If not, see <http://www.gnu.org/licenses/>.
!

module gpu_clinfc

use errors
use fc_cpsolv
use variables_global
use, intrinsic :: iso_c_binding

implicit none

  !! Define GPU Variables
  integer(kind=4), dimension(:), allocatable :: FILL_LI, FILL_LK, FILL_MINUS, FSBS, FORDER, CORDER
  double precision, dimension(:), allocatable :: BET, BETO, VV, VRES, FILL !working arrays
  double precision, dimension(:,:), pointer :: h_R
  double precision, dimension(:), pointer :: d_A
  type(C_PTR) :: cptr_d_li, cptr_d_lk, cptr_d_ae, cptr_d_aw, cptr_d_an, cptr_d_forder, cptr_d_corder
  type(C_PTR) :: cptr_d_as, cptr_d_at, cptr_d_ab, cptr_d_ap, cptr_d_q
  type(C_PTR) :: cptr_d_pk, cptr_d_zk, cptr_d_d, cptr_d_res
  type(C_PTR) :: cptr_d_uk, cptr_d_vk, cptr_d_lb, cptr_d_lw, cptr_d_ls
  type(C_PTR) :: cptr_d_lpr, cptr_d_un, cptr_d_ue, cptr_d_ut, cptr_d_reso
  type(C_PTR) :: cptr_d_fi, cptr_d_bet, cptr_d_beto, cptr_d_vv, cptr_d_vres
  type(C_PTR) :: cptr_queue, cptr_queue2, cptr_d_ijk
  type(C_PTR) :: cptr_d_rsm, cptr_d_resmax, cptr_d_om, cptr_d_alf, cptr_d_gam
  type(C_PTR) :: cptr_d_resl, cptr_d_ukreso, cptr_d_betol, cptr_d_res0
  type(C_PTR) :: cptr_d_reduce_dt1, cptr_d_reduce_dt2, cptr_d_zkt, cptr_d_block_finish
  type(C_PTR) :: cptr_d_nij, cptr_d_nj, cptr_d_njm, cptr_d_nxyz
  integer(C_SIZE_T), parameter :: sizeof_integer = 4
  integer(C_SIZE_T), parameter :: sizeof_double = 8
  integer(C_SIZE_T), parameter :: sizeof_ptr = 8
  integer(C_SIZE_T), dimension(3) :: localworksize, globalworksize, localworksize_reduce, localworksize_sub
  integer(C_SIZE_T), dimension(3) :: globalworksize_reduce, globalworksize_sub
  integer(kind=8), target :: queue, queue2 !cuda dim(7)
  integer(C_SIZE_T) :: blocks, threads, threads_reduce,forder_size
  integer, parameter :: fp_kind = kind(0.0d0) ! Double precision

  integer :: ttime=-4

  integer :: pinned=0 !pinned memory 0=no, 1=yes

!include 'magma.h'

contains
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! Gpu_infc_init - Allocates gpu variables
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine gpu_clinfc_init

   integer clinterface_init, clinterface_get_devices, clinterface_queue_create, clinterface_malloc
   external clinterface_init, clinterface_get_devices, clinterface_queue_create, clinterface_malloc

   integer clinterface_tp_uint, clinterface_buildkernels, clinterface_malloc_pinned, clinterface_malloc_mapped
   external clinterface_tp_uint, clinterface_buildkernels, clinterface_malloc_pinned, clinterface_malloc_mapped

   integer clinterface_clretainmemobject
   external clinterface_clretainmemobject

   integer clinterface_malloc_readonly, clinterface_malloc_mapped_readonly
   external clinterface_malloc_readonly, clinterface_malloc_mapped_readonly

   integer clinterface_dsetmatrix, clinterface_dsetmatrix_nb
   external clinterface_dsetmatrix, clinterface_dsetmatrix_nb

   integer clinterface_isetmatrix, clinterface_isetmatrix_nb, clinterface_clbarrier
   external clinterface_isetmatrix, clinterface_isetmatrix_nb, clinterface_clbarrier

   external clinterface_clflush, clinterface_clfinish
   integer clinterface_clfinish

   integer clinterface_queue_create_async
   external clinterface_queue_create_async

   !integer(kind=8), intent(in) :: size_fp
   !integer(kind=8), intent(inout) :: lda, ldda

   integer :: i,j,k,ct,bx,by,bz,add
   integer(kind=4) :: err, rank_gpu
   character*100 :: var
   integer :: nxyz, nxyz_reduce
   integer(C_SIZE_T) :: sizegp, sizetotal, sizearray
   integer(C_INT) :: stat

   integer(kind=4), target, dimension(10) :: dev
   type(C_PTR) :: cptr_dev
   integer(kind=4) :: ndev
   integer(kind=4) :: zero = 0, one = 1 !32bit integers

   !! Allocate GPU variables
   nxyz=(nxmax-1)*(nymax-1)*(nzmax-1)
   nxyz_reduce=(nzmax-1)*(nxmax-1)*(nymax-1)

   !GPU parameters - CGSTAB Kernel
   !In this version we will use the 1D version of the GPU Code - this simplifies the approach
   threads = 16 !number of threads - depends on gpu type, usually 32 - should be a power of 2 for memory access reasons
   localworksize(1) = threads !this local work size should be optimized for your gpu
   localworksize(2) = 0
   localworksize(3) = 0
   globalworksize(1) = (((nxyz)+(localworksize(1)-1))/localworksize(1))*localworksize(1)
   globalworksize(2) = 1
   globalworksize(3) = 1

   !GPU parameters - Reduction Kernel
   threads_reduce = 16 !number of threads - depends on gpu type, usually 32 - should be a power of 2 for memory access reasons
   localworksize_reduce(1) = threads_reduce !this local work size should be optimized for your gpu
   localworksize_reduce(2) = 0
   localworksize_reduce(3) = 0
   globalworksize_reduce(1) = (((nxyz_reduce)+(localworksize_reduce(1)-1))/localworksize_reduce(1))*localworksize_reduce(1)
   globalworksize_reduce(2) = 1
   globalworksize_reduce(3) = 1

   !write(*,*) 'localworksize(:) =',localworksize(:)
   !write(*,*) 'globalworksize(:) =',globalworksize(:),(nxmax-1),(nymax-1),(nzmax-1)

   write(*,*) ' '
   write(*,*) ' ** Determining Layout on GPU Cards - CGSTAB Routine'
   write(*,*) '----------------------------------------------------------------------'
   write(*,*) ' Number of Threads = ',localworksize(1)
   write(*,*) ' Number of Blocks  = ',globalworksize(1)/localworksize(1)
   write(*,*) ' Total Grid Size   = ',globalworksize(1)
   write(*,*) '----------------------------------------------------------------------'
   write(*,*) ' '
   write(*,*) ' ** Determining Layout on GPU Cards - Reduction Routine'
   write(*,*) '----------------------------------------------------------------------'
   write(*,*) ' Number of Threads = ',localworksize_reduce(1)
   write(*,*) ' Number of Blocks  = ',globalworksize_reduce(1)/localworksize_reduce(1)
   write(*,*) ' Total Grid Size   = ',globalworksize_reduce(1)
   write(*,*) '----------------------------------------------------------------------'
 
   dev(:) = 0
   write(*,*) ' '
   write(*,*) ' ** Initializing GPU Cards'

   !!Initialize AMDblas
   rank_gpu=0 !gpu number to select
   stat = clinterface_init(rank_gpu)
   if(stat .ne. 0 ) then
     write(var,*) 'Gpu Init Failed:',stat
     call errors_clblas(stat,var)
   end if

   !get devices - will select device based on mpi rank
   cptr_dev = C_LOC(dev)
   stat = clinterface_get_devices(dev, 10, ndev);
   if(stat .ne. 0 ) then
     write(var,*) 'Get Devices Failed:',stat
     call errors_clblas(stat,var)
   end if

   !in-order queue for kernel execution
   cptr_queue = C_LOC(queue)
   stat = clinterface_queue_create(dev(rank_gpu+1), cptr_queue);
   !out-order queue for transfers
   cptr_queue2 = C_LOC(queue2)
   stat = clinterface_queue_create(dev(rank_gpu+1), cptr_queue2);
   !stat = clinterface_queue_create_async(dev(rank_gpu+1), cptr_queue2);
   if(stat .ne. 0 ) then
     write(var,*) 'clCreateCommandQueue failed:',stat
     call errors_clblas(stat,var)
   end if

   !!Initialize Magma
   write(*,*) '    Building Cl Kernels'
   stat = clinterface_buildkernels(dev(rank_gpu+1))
   write(var,*) 'clcgstab.cl',stat
   call errors_clbuild(stat,var)

   !! Allocate GPU device memory
   write(*,*) '    Initializing Working Arrays'
   allocate(BET(nxyz),BETO(nxyz),VV(nxyz),VRES(nxyz), FILL(globalworksize(1)*globalworksize(2)*globalworksize(3)), stat=err)
   allocate(FILL_LI(globalworksize(1)), FILL_LK(globalworksize(3)), stat=err)
   FILL(:) = 0; FILL_LI(:) = 0; FILL_LK(:) = 0; 
   write(*,*) '    Allocating GPU Local Memory'
   sizetotal=0

   !integer arrays on GPU
   sizegp = (globalworksize(1)*globalworksize(2)*globalworksize(3))*sizeof_integer; sizetotal=sizegp+sizetotal
   stat = clinterface_malloc(cptr_d_ijk,sizegp)
   sizegp = (globalworksize(1)*globalworksize(2)*globalworksize(3)/localworksize(1))*sizeof_integer; sizetotal=sizegp+sizetotal
   sizearray = (globalworksize(1)*globalworksize(2)*globalworksize(3))
   stat = clinterface_isetmatrix_nb(sizearray,1,FILL_LI,zero,sizearray,cptr_d_ijk,zero,sizearray,cptr_queue2)

   !set predefined array for forward and backsubstitution directions
   !this is much easier to do upfront than figure this out on gpus.
   localworksize_sub(1) = 16 !this local work size should be optimized for your gpu
   localworksize_sub(2) = 0
   localworksize_sub(3) = 0
   bx=(((nxmax-1)+(1-1))/1)*1
   by=(((nymax-1)+(localworksize_sub(1)-1))/localworksize_sub(1))
   bz=(((nzmax-1)+(1-1))/1)*1
   allocate(FSBS(bx*by*bz),FORDER(bx*by*bz),FILL_MINUS(bx*by*bz), stat=err)

   call gpu_clinterface1D_forder(nxmax,nymax,nzmax)

   sizegp = bx*by*bz*sizeof_integer; sizetotal=sizegp*2+sizetotal
   sizearray = bx*by*bz
   forder_size = bx*by*bz
   stat = clinterface_malloc(cptr_d_forder,sizegp)
   stat = clinterface_malloc(cptr_d_block_finish,sizegp)
   stat = clinterface_isetmatrix_nb(sizearray,1,FORDER,zero,sizearray,cptr_d_forder,zero,sizearray,cptr_queue2)
   stat = clinterface_isetmatrix_nb(sizearray,1,FILL_MINUS,zero,sizearray,cptr_d_block_finish,zero,sizearray,cptr_queue2)

   if (stat .ne. 0) then
      write(*,*) "Device Memory Allocation Failed, Stat = ",stat
      stop
   endif

   !integer array
   allocate(CORDER(nxyz), stat=err)
   sizegp = nxyz*sizeof_integer;
   if(pinned .eq. 0) then 
     stat = clinterface_malloc(cptr_d_corder,sizegp)
   else
     stat = clinterface_malloc_mapped(cptr_d_corder,sizegp,CORDER)
   endif

   !double arrays on GPU
   sizegp = (globalworksize(1)*globalworksize(2)*globalworksize(3))*sizeof_double; sizetotal=21*sizegp+sizetotal
   sizearray = (globalworksize(1)*globalworksize(2)*globalworksize(3))
   if(pinned .eq. 0) then
     stat = clinterface_malloc_readonly(cptr_d_ae,sizegp) !1
     stat = clinterface_malloc_readonly(cptr_d_aw,sizegp) !2
     stat = clinterface_malloc_readonly(cptr_d_an,sizegp) !3
     stat = clinterface_malloc_readonly(cptr_d_as,sizegp) !4
     stat = clinterface_malloc_readonly(cptr_d_at,sizegp) !5
     stat = clinterface_malloc_readonly(cptr_d_ab,sizegp) !6
     stat = clinterface_malloc_readonly(cptr_d_ap,sizegp) !7
     stat = clinterface_malloc_readonly(cptr_d_q,sizegp)  !8
     stat = clinterface_malloc_readonly(cptr_d_d,sizegp)  !9

     stat = clinterface_malloc_readonly(cptr_d_nij,sizeof_integer)  !9a
     stat = clinterface_malloc_readonly(cptr_d_nj ,sizeof_integer)  !9b
     stat = clinterface_malloc_readonly(cptr_d_njm,sizeof_integer)  !9c
     stat = clinterface_malloc_readonly(cptr_d_nxyz,sizeof_integer) !9d

     stat = clinterface_malloc_readonly(cptr_d_alf,sizeof_double)  !9e
     stat = clinterface_malloc_readonly(cptr_d_gam,sizeof_double)  !9f
     stat = clinterface_malloc_readonly(cptr_d_om,sizeof_double)  !9g

     write (*,*) '    Allocating Device Read-Only Memory = ',((9*sizegp+4*sizeof_integer)/1E6),'Mbytes on GPU'
   else
     stat = clinterface_malloc_mapped_readonly(cptr_d_ae,sizegp,AE) !1
     stat = clinterface_malloc_mapped_readonly(cptr_d_aw,sizegp,AW) !2
     stat = clinterface_malloc_mapped_readonly(cptr_d_an,sizegp,AN) !3
     stat = clinterface_malloc_mapped_readonly(cptr_d_as,sizegp,AS) !4
     stat = clinterface_malloc_mapped_readonly(cptr_d_at,sizegp,AT) !5
     stat = clinterface_malloc_mapped_readonly(cptr_d_ab,sizegp,AB) !6
     stat = clinterface_malloc_mapped_readonly(cptr_d_ap,sizegp,AP) !7
     stat = clinterface_malloc_mapped_readonly(cptr_d_q,sizegp,Q)   !8
     stat = clinterface_malloc_mapped_readonly(cptr_d_d,sizegp,D)   !9

     stat = clinterface_malloc_mapped_readonly(cptr_d_nij ,sizeof_integer,nij)  !9a
     stat = clinterface_malloc_mapped_readonly(cptr_d_nj  ,sizeof_integer,nj)   !9b
     stat = clinterface_malloc_mapped_readonly(cptr_d_njm ,sizeof_integer,njm)  !9c
     stat = clinterface_malloc_mapped_readonly(cptr_d_nxyz,sizeof_integer,nxyz) !9d
   endif
   if (stat .ne. 0) then
      write(*,*) "Device Memory Allocation Failed, Stat = ",stat
      stop
   endif

   if(pinned .eq. 0) then
     stat = clinterface_malloc(cptr_d_pk,sizegp)   !10
     stat = clinterface_malloc(cptr_d_zk,sizegp)   !11
     stat = clinterface_malloc(cptr_d_zkt,sizegp)  !11a
     stat = clinterface_malloc(cptr_d_res,sizegp)  !12
     stat = clinterface_malloc(cptr_d_uk,sizegp)   !13
     stat = clinterface_malloc(cptr_d_vk,sizegp)   !14
     stat = clinterface_malloc(cptr_d_reso,sizegp) !15
     stat = clinterface_malloc(cptr_d_bet,sizegp)  !17
     stat = clinterface_malloc(cptr_d_vv,sizegp)   !19
     stat = clinterface_malloc(cptr_d_vres,sizegp) !20

     !these are the only two that are really doing transfer
     stat = clinterface_malloc(cptr_d_fi  ,sizegp) !16
     stat = clinterface_malloc(cptr_d_beto,sizegp) !18
     write (*,*) '    Allocating Device Memory = ',(12*sizegp/1E6),'Mbytes on GPU'
   endif

   if (stat .ne. 0) then
      write(*,*) "Device Memory Allocation Failed, Stat = ",stat
      stop
   endif

   write (*,*) '    Allocating Total Device Memory = ',sizetotal/1E6,'Mbytes on GPU'

   if(pinned .eq. 0) then
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_ae  ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_aw  ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_an  ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_as  ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_at  ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_ab  ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_ap  ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_q   ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_d   ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
   endif
   if (stat .ne. 0) then
      write(*,*) "Read-Only Memory Initilization Failed, Stat = ",stat
      stop
   endif
   if(pinned .eq. 0) then
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_pk  ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_zk  ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_zkt ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_res ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_vk  ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_uk  ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_reso,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_bet ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_vv  ,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(sizearray,1,FILL,zero,sizearray,cptr_d_vres,zero,sizearray,cptr_queue2) ! Send vector to gpu memory
   end if
   !stat = clinterface_clfinish(cptr_queue2) !flush gpu command queues
   if (stat .ne. 0) then
      write(*,*) "Memory Initilization Failed, Stat = ",stat
      stop
   endif

   write(*,*) '    GPUs Allocated'

   call gpu_clinfc_cgstab_init

   write(*,*) '    GPUs Arguments Initialized'

end subroutine gpu_clinfc_init

subroutine gpu_clinterface1D_forder(nx,ny,nz)

   integer :: bx, by, bz, i, j, k, ijk, n, add, ct
   integer, intent(in) :: nx,ny,nz

   bx=(((nxmax-1)+(1-1))/1)*1
   by=(((nymax-1)+(localworksize_sub(1)-1))/localworksize_sub(1))
   bz=(((nzmax-1)+(1-1))/1)*1

   FSBS(:)=globalworksize(1); FORDER(:)=globalworksize(1); FILL_MINUS(:)=-1
   globalworksize_sub(1)=bx*by*bz*localworksize_sub(1)
   globalworksize_sub(2)=1
   globalworksize_sub(3)=1

   add=1
   do i=1,bx*bz
     do j=1,by
       !FSBS(j+(i-1)*by)=j+(i-1)
       FSBS(j+(i-1)*by)=add
       add=add+1
     end do
   end do

   add=0
   do ct=1,bx*by*bz
     do i=1,bx*bz
       do j=1,by
         if(FSBS(j+(i-1)*by) .eq. ct) then
           FORDER(j+(i-1)*by)=add
           add=add+1
         end if
       end do
     end do
   end do

end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! CGSTAB_GPU - Run CGSTAB on GPU.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine CGSTAB_GPU(MAXIT)

   integer, intent(in) :: MAXIT !,CGT 
   integer :: CGT =1

   !write(*,*) '    GPUs Set'
   call gpu_clinfc_set
   !write(*,*) '    GPUs Run'
   if(CGT .eq. 1) then
     call gpu_clinfc_cgstab(MAXIT) !CGSTAB
   elseif(CGT .eq. 2) then
     !call gpu_clinfc_icgstab(MAXIT) !ICGSTAB
   endif
   !write(*,*) '    GPUs Get'
   call gpu_clinfc_get

end subroutine CGSTAB_GPU

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! gpu_clinfc_set - Set arrays on GPUs
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine gpu_clinfc_set

   use fc_cpsolv

   integer clinterface_dsetmatrix, clinterface_dsetmatrix_nb, clinterface_clbarrier
   external clinterface_dsetmatrix, clinterface_dsetmatrix_nb, clinterface_clbarrier

   integer clinterface_isetmatrix, clinterface_isetmatrix_nb
   external clinterface_isetmatrix, clinterface_isetmatrix_nb

   external clinterface_clflush, clinterface_clfinish

   integer(C_INT) :: stat
   integer(kind=4) :: zero = 0, one = 1 !32bit integers
   integer :: nxyz
   real(kind=4) :: t_gpu_end,t_gpu_start

   nxyz=(nx-1)*(ny-1)*(nz-1)

   if(pinned .eq. 0) then
     call cpu_time(t_gpu_start)
     stat = clinterface_isetmatrix_nb(1,1,nxyz,zero,1,cptr_d_nxyz,zero,1,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_isetmatrix_nb(1,1,NIJ ,zero,1,cptr_d_nij ,zero,1,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_isetmatrix_nb(1,1,NJ  ,zero,1,cptr_d_nj  ,zero,1,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_isetmatrix_nb(1,1,NJM ,zero,1,cptr_d_njm ,zero,1,cptr_queue2) ! Send vector to gpu memory

     stat = clinterface_isetmatrix_nb(nxyz,1,IJKS,zero,nxyz,cptr_d_ijk,zero,nxyz,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(nxyz,1,AE  ,zero,nxyz,cptr_d_ae ,zero,nxyz,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(nxyz,1,AW  ,zero,nxyz,cptr_d_aw ,zero,nxyz,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(nxyz,1,AN  ,zero,nxyz,cptr_d_an ,zero,nxyz,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(nxyz,1,AS  ,zero,nxyz,cptr_d_as ,zero,nxyz,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(nxyz,1,AT  ,zero,nxyz,cptr_d_at ,zero,nxyz,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(nxyz,1,AB  ,zero,nxyz,cptr_d_ab ,zero,nxyz,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(nxyz,1,AP  ,zero,nxyz,cptr_d_ap ,zero,nxyz,cptr_queue2) ! Send vector to gpu memory
     stat = clinterface_dsetmatrix_nb(nxyz,1,Q   ,zero,nxyz,cptr_d_q  ,zero,nxyz,cptr_queue2) ! Send vector to gpu memory
     !stat = clinterface_dsetmatrix(nxyz,1,D ,zero,nxyz,cptr_d_d ,zero,nxyz,cptr_queue) ! Send vector to gpu memory
     ! T is the array with the results
     stat = clinterface_dsetmatrix_nb(nxyz,1,T   ,zero,nxyz,cptr_d_fi ,zero,nxyz,cptr_queue2) ! Send vector to gpu memory
     if (stat .ne. 0) then
        write(*,*) "Matrix Set Failed, Stat = ",stat
        stop
     endif
     !call clinterface_clfinish(cptr_queue2)
     !stat = clinterface_clbarrier(cptr_queue2) !flush gpu command queues
     call cpu_time(t_gpu_end)
     write(*,*) 'GPU Set Time = ',t_gpu_end-t_gpu_start,'sec'
   endif


end subroutine

subroutine gpu_clinfc_get

   use fc_cpsolv

   integer clinterface_dgetmatrix, clinterface_dgetmatrix_nb, clinterface_clbarrier
   external clinterface_dgetmatrix, clinterface_dgetmatrix_nb, clinterface_clbarrier

   external clinterface_clflush, clinterface_clfinish
   integer clinterface_clfinish

   integer :: nxyz
   integer(C_INT) :: stat
   integer(kind=4) :: zero = 0, one = 1 !32bit integers
   nxyz=(nx-1)*(ny-1)*(nz-1)

   if(pinned .eq. 0) then
     stat = clinterface_dgetmatrix_nb(nxyz,1,cptr_d_fi,zero,nxyz,T,zero,nxyz,cptr_queue2)
     stat = clinterface_dgetmatrix_nb(nxyz,1,cptr_d_res,zero,nxyz,RES,zero,nxyz,cptr_queue2)
     !stat = clinterface_dgetmatrix(nxyz,1,cptr_d_d,zero,nxyz,D,zero,nxyz,cptr_queue)
     stat = clinterface_clfinish(cptr_queue2) !flush gpu command queues
     if (stat .ne. 0) then
        write(*,*) "Matrix Get Failed, Stat = ",stat
        stop
     endif
   endif
   call clinterface_clflush(cptr_queue) !flush gpu command queues



end subroutine

subroutine gpu_clinfc_cgstab_init

   integer clinterface_clsetkernelarg
   external clinterface_clsetkernelarg
   integer clinterface_clsetkernelarg_ptr, clinterface_clsetkernelarg_shmem
   external clinterface_clsetkernelarg_ptr, clinterface_clsetkernelarg_shmem

   integer :: nxyz, nxyz_reduce, nxyz_range
   integer(C_INT) :: stat
   integer(C_SIZE_T) :: shmem_size, shmem_sub_size

   nxyz=(nxmax-1)*(nymax-1)*(nzmax-1) !size of arrays
   nxyz_reduce=(nxmax-1)*(nymax-1)*(nzmax-1)

   ! Kernel Num, Kernel Name
   !          0->?, clcgstab1D_kernel
   !          1->22, clcgstab1D_reduce
   !          2->23, clcgstab1D_reduce_abs
   !          3->27, clcgstab1D_zer
   !          4->28, clcgstab1D_irv
   !          5->29, clcgstab1D_pmd
   !          6->30, clcgstab1D_iwa
   !          7->31, clcgstab1D_cbo
   !          8->32, clcgstab1D_cpk
   !          9->33, clcgstab1D_sfs
   !         10->34, clcgstab1D_sbs
   !         11->35, clcgstab1D_cuk
   !         12->36, clcgstab1D_ruk
   !         13->37, clcgstab1D_ufr
   !         14->38, clcgstab1D_smy
   !         15->39, clcgstab1D_cav
   !         16->40, clcgstab1D_caf
   !         17->41, clcgstab1D_cfr
   !         18->42, clcgstab1D_sfs2
   !         19->43, clcgstab1D_smy2
   !         20->44, clcgstab1D_sbs2
   !         21->45, clcgstab1D_pmd2

   !clcgstab_zer 27
   stat = clinterface_clsetkernelarg_ptr(27, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(27, 1, sizeof_ptr, cptr_d_pk)
   stat = clinterface_clsetkernelarg_ptr(27, 2, sizeof_ptr, cptr_d_zk)
   stat = clinterface_clsetkernelarg_ptr(27, 3, sizeof_ptr, cptr_d_zkt)
   stat = clinterface_clsetkernelarg_ptr(27, 4, sizeof_ptr, cptr_d_vk)
   stat = clinterface_clsetkernelarg_ptr(27, 5, sizeof_ptr, cptr_d_uk)
   stat = clinterface_clsetkernelarg_ptr(27, 6, sizeof_ptr, cptr_d_res)
   stat = clinterface_clsetkernelarg_ptr(27, 7, sizeof_ptr, cptr_d_reso)
   stat = clinterface_clsetkernelarg_ptr(27, 8, sizeof_ptr, cptr_d_bet)
   stat = clinterface_clsetkernelarg_ptr(27, 9, sizeof_ptr, cptr_d_beto)
   stat = clinterface_clsetkernelarg_ptr(27, 10,sizeof_ptr, cptr_d_vv)
   stat = clinterface_clsetkernelarg_ptr(27, 11,sizeof_ptr, cptr_d_vres)
   stat = clinterface_clsetkernelarg_ptr(27, 12,sizeof_ptr, cptr_d_d)
   stat = clinterface_clsetkernelarg_ptr(27, 13,sizeof_ptr, cptr_d_corder)

   !clcgstab_irv 28
   stat = clinterface_clsetkernelarg_ptr(28, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(28, 1, sizeof_ptr, cptr_d_nij)
   stat = clinterface_clsetkernelarg_ptr(28, 2, sizeof_ptr, cptr_d_nj)
   stat = clinterface_clsetkernelarg_ptr(28, 3, sizeof_ptr, cptr_d_res)
   stat = clinterface_clsetkernelarg_ptr(28, 4, sizeof_ptr, cptr_d_q)
   stat = clinterface_clsetkernelarg_ptr(28, 5, sizeof_ptr, cptr_d_ap)
   stat = clinterface_clsetkernelarg_ptr(28, 6, sizeof_ptr, cptr_d_ae)
   stat = clinterface_clsetkernelarg_ptr(28, 7, sizeof_ptr, cptr_d_aw)
   stat = clinterface_clsetkernelarg_ptr(28, 8, sizeof_ptr, cptr_d_an)
   stat = clinterface_clsetkernelarg_ptr(28, 9, sizeof_ptr, cptr_d_as)
   stat = clinterface_clsetkernelarg_ptr(28, 10,sizeof_ptr, cptr_d_at)
   stat = clinterface_clsetkernelarg_ptr(28, 11,sizeof_ptr, cptr_d_ab)
   stat = clinterface_clsetkernelarg_ptr(28, 12,sizeof_ptr, cptr_d_fi)
   stat = clinterface_clsetkernelarg_ptr(28, 13,sizeof_ptr, cptr_d_corder)

   !clcreduce_abs
   stat = clinterface_clsetkernelarg_ptr(22, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(22, 1, sizeof_ptr, cptr_d_corder)
   stat = clinterface_clsetkernelarg_ptr(23, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(23, 1, sizeof_ptr, cptr_d_corder)
   shmem_size = threads_reduce*sizeof_double
   stat = clinterface_clsetkernelarg_shmem(22, 4, shmem_size) !shared memory
   stat = clinterface_clsetkernelarg_shmem(23, 4, shmem_size) !shared memory

   !clcgstab_iwa 30
   stat = clinterface_clsetkernelarg_ptr(30, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(30, 1, sizeof_ptr, cptr_d_pk)
   stat = clinterface_clsetkernelarg_ptr(30, 2, sizeof_ptr, cptr_d_uk)
   stat = clinterface_clsetkernelarg_ptr(30, 3, sizeof_ptr, cptr_d_zk)
   stat = clinterface_clsetkernelarg_ptr(30, 4, sizeof_ptr, cptr_d_vk)
   stat = clinterface_clsetkernelarg_ptr(30, 5, sizeof_ptr, cptr_d_res)
   stat = clinterface_clsetkernelarg_ptr(30, 6, sizeof_ptr, cptr_d_reso)
   stat = clinterface_clsetkernelarg_ptr(30, 7, sizeof_ptr, cptr_d_corder)

   !clcgstab1D_cbo
   stat = clinterface_clsetkernelarg_ptr(31, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(31, 1, sizeof_ptr, cptr_d_bet)
   stat = clinterface_clsetkernelarg_ptr(31, 2, sizeof_ptr, cptr_d_res)
   stat = clinterface_clsetkernelarg_ptr(31, 3, sizeof_ptr, cptr_d_reso)
   stat = clinterface_clsetkernelarg_ptr(31, 4, sizeof_ptr, cptr_d_corder)

   !clcgstab1D_cpk
   stat = clinterface_clsetkernelarg_ptr(32, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(32, 1, sizeof_ptr, cptr_d_pk)
   stat = clinterface_clsetkernelarg_ptr(32, 2, sizeof_ptr, cptr_d_res)
   stat = clinterface_clsetkernelarg_ptr(32, 3, sizeof_ptr, cptr_d_uk)
   stat = clinterface_clsetkernelarg_ptr(32, 4, sizeof_ptr, cptr_d_corder)
   stat = clinterface_clsetkernelarg_ptr(32, 5, sizeof_ptr, cptr_d_om)
   stat = clinterface_clsetkernelarg_ptr(32, 6, sizeof_ptr, cptr_d_alf)

   !clcgstab1D_sfs
   stat = clinterface_clsetkernelarg_ptr(33, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(33, 1, sizeof_ptr, cptr_d_njm)
   stat = clinterface_clsetkernelarg_ptr(33, 2, sizeof_ptr, cptr_d_nij)
   stat = clinterface_clsetkernelarg_ptr(33, 3, sizeof_ptr, cptr_d_nj)
   stat = clinterface_clsetkernelarg_ptr(33, 4, sizeof_ptr, cptr_d_pk)
   stat = clinterface_clsetkernelarg_ptr(33, 5, sizeof_ptr, cptr_d_aw)
   stat = clinterface_clsetkernelarg_ptr(33, 6, sizeof_ptr, cptr_d_zk)
   stat = clinterface_clsetkernelarg_ptr(33, 7, sizeof_ptr, cptr_d_as)
   stat = clinterface_clsetkernelarg_ptr(33, 8, sizeof_ptr, cptr_d_ab)
   stat = clinterface_clsetkernelarg_ptr(33, 9, sizeof_ptr, cptr_d_d)
   stat = clinterface_clsetkernelarg_ptr(33, 10,sizeof_ptr, cptr_d_block_finish)
   stat = clinterface_clsetkernelarg_ptr(33, 11,sizeof_ptr, cptr_d_forder)
   shmem_sub_size = localworksize_sub(1)*sizeof_double !hardcode 16 threads
   stat = clinterface_clsetkernelarg_shmem(33, 12, shmem_sub_size) !shared memory
   stat = clinterface_clsetkernelarg_shmem(33, 13, shmem_sub_size) !shared memory
   stat = clinterface_clsetkernelarg_shmem(33, 14, shmem_sub_size) !shared memory
   stat = clinterface_clsetkernelarg_shmem(33, 15, shmem_sub_size) !shared memory
   shmem_sub_size = localworksize_sub(1)*sizeof_integer !hardcode 17 threads
   stat = clinterface_clsetkernelarg_shmem(33, 16, shmem_sub_size) !shared memory

   !clcgstab1D_sfs_init
   stat = clinterface_clsetkernelarg_ptr(46, 0, sizeof_ptr, cptr_d_block_finish)

   !clcgstab1D_sfs2
   stat = clinterface_clsetkernelarg_ptr(42, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(42, 1, sizeof_ptr, cptr_d_zk)
   stat = clinterface_clsetkernelarg_ptr(42, 2, sizeof_ptr, cptr_d_d)
   stat = clinterface_clsetkernelarg_ptr(42, 3, sizeof_ptr, cptr_d_corder)

   !clcgstab1D_sbs
   stat = clinterface_clsetkernelarg_ptr(34, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(34, 1, sizeof_ptr, cptr_d_njm)
   stat = clinterface_clsetkernelarg_ptr(34, 2, sizeof_ptr, cptr_d_nij)
   stat = clinterface_clsetkernelarg_ptr(34, 3, sizeof_ptr, cptr_d_nj)
   stat = clinterface_clsetkernelarg_ptr(34, 4, sizeof_ptr, cptr_d_pk)
   stat = clinterface_clsetkernelarg_ptr(34, 5, sizeof_ptr, cptr_d_ae)
   stat = clinterface_clsetkernelarg_ptr(34, 6, sizeof_ptr, cptr_d_zk)
   stat = clinterface_clsetkernelarg_ptr(34, 7, sizeof_ptr, cptr_d_an)
   stat = clinterface_clsetkernelarg_ptr(34, 8, sizeof_ptr, cptr_d_at)
   stat = clinterface_clsetkernelarg_ptr(34, 9, sizeof_ptr, cptr_d_d)
   stat = clinterface_clsetkernelarg_ptr(34, 10,sizeof_ptr, cptr_d_block_finish)
   stat = clinterface_clsetkernelarg_ptr(34, 11,sizeof_ptr, cptr_d_forder)
   shmem_sub_size = localworksize_sub(1)*sizeof_double !hardcode 16 threads
   stat = clinterface_clsetkernelarg_shmem(34, 12, shmem_sub_size) !shared memory
   stat = clinterface_clsetkernelarg_shmem(34, 13, shmem_sub_size) !shared memory
   stat = clinterface_clsetkernelarg_shmem(34, 14, shmem_sub_size) !shared memory
   stat = clinterface_clsetkernelarg_shmem(34, 15, shmem_sub_size) !shared memory
   shmem_sub_size = localworksize_sub(1)*sizeof_integer !hardcode 17 threads
   stat = clinterface_clsetkernelarg_shmem(34, 16, shmem_sub_size) !shared memory

   !clcgstab1D_sbs2
   !stat = clinterface_clsetkernelarg_ptr(44, 0, sizeof_integer, nxyz)
   !stat = clinterface_clsetkernelarg_ptr(44, 1, sizeof_ptr, cptr_d_zk)
   !stat = clinterface_clsetkernelarg_ptr(44, 2, sizeof_ptr, cptr_d_zkt)
   !stat = clinterface_clsetkernelarg_ptr(44, 3, sizeof_ptr, cptr_d_corder)

   !clcgstab1D_cuk
   stat = clinterface_clsetkernelarg_ptr(35, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(35, 1, sizeof_ptr, cptr_d_nij)
   stat = clinterface_clsetkernelarg_ptr(35, 2, sizeof_ptr, cptr_d_nj)
   stat = clinterface_clsetkernelarg_ptr(35, 3, sizeof_ptr, cptr_d_uk)
   stat = clinterface_clsetkernelarg_ptr(35, 4, sizeof_ptr, cptr_d_ap)
   stat = clinterface_clsetkernelarg_ptr(35, 5, sizeof_ptr, cptr_d_zk)
   stat = clinterface_clsetkernelarg_ptr(35, 6, sizeof_ptr, cptr_d_ae)
   stat = clinterface_clsetkernelarg_ptr(35, 7, sizeof_ptr, cptr_d_aw)
   stat = clinterface_clsetkernelarg_ptr(35, 8, sizeof_ptr, cptr_d_an)
   stat = clinterface_clsetkernelarg_ptr(35, 9, sizeof_ptr, cptr_d_as)
   stat = clinterface_clsetkernelarg_ptr(35, 10,sizeof_ptr, cptr_d_at)
   stat = clinterface_clsetkernelarg_ptr(35, 11,sizeof_ptr, cptr_d_ab)
   stat = clinterface_clsetkernelarg_ptr(35, 12,sizeof_ptr, cptr_d_corder)

   !clcgstab1D_ruk
   stat = clinterface_clsetkernelarg_ptr(36, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(36, 1, sizeof_ptr, cptr_d_vv)
   stat = clinterface_clsetkernelarg_ptr(36, 2, sizeof_ptr, cptr_d_reso)
   stat = clinterface_clsetkernelarg_ptr(36, 3, sizeof_ptr, cptr_d_uk)
   stat = clinterface_clsetkernelarg_ptr(36, 4, sizeof_ptr, cptr_d_corder)

   !clcgstab1D_ufr
   stat = clinterface_clsetkernelarg_ptr(37, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(37, 1, sizeof_ptr, cptr_d_zk)
   stat = clinterface_clsetkernelarg_ptr(37, 2, sizeof_ptr, cptr_d_res)
   stat = clinterface_clsetkernelarg_ptr(37, 3, sizeof_ptr, cptr_d_fi)
   stat = clinterface_clsetkernelarg_ptr(37, 4, sizeof_ptr, cptr_d_uk)
   stat = clinterface_clsetkernelarg_ptr(37, 5, sizeof_ptr, cptr_d_corder)
   stat = clinterface_clsetkernelarg_ptr(37, 6, sizeof_ptr, cptr_d_gam)

   !clcgstab1D_smy
   stat = clinterface_clsetkernelarg_ptr(38, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(38, 1, sizeof_ptr, cptr_d_njm)
   stat = clinterface_clsetkernelarg_ptr(38, 2, sizeof_ptr, cptr_d_nij)
   stat = clinterface_clsetkernelarg_ptr(38, 3, sizeof_ptr, cptr_d_nj)
   stat = clinterface_clsetkernelarg_ptr(38, 4, sizeof_ptr, cptr_d_res)
   stat = clinterface_clsetkernelarg_ptr(38, 5, sizeof_ptr, cptr_d_zk)
   stat = clinterface_clsetkernelarg_ptr(38, 6, sizeof_ptr, cptr_d_aw)
   stat = clinterface_clsetkernelarg_ptr(38, 7, sizeof_ptr, cptr_d_as)
   stat = clinterface_clsetkernelarg_ptr(38, 8, sizeof_ptr, cptr_d_ab)
   stat = clinterface_clsetkernelarg_ptr(38, 9, sizeof_ptr, cptr_d_d)
   stat = clinterface_clsetkernelarg_ptr(38, 10,sizeof_ptr, cptr_d_block_finish)
   stat = clinterface_clsetkernelarg_ptr(38, 11,sizeof_ptr, cptr_d_forder)
   shmem_sub_size = localworksize_sub(1)*sizeof_double !hardcode 16 threads
   stat = clinterface_clsetkernelarg_shmem(38, 12, shmem_sub_size) !shared memory
   stat = clinterface_clsetkernelarg_shmem(38, 13, shmem_sub_size) !shared memory
   stat = clinterface_clsetkernelarg_shmem(38, 14, shmem_sub_size) !shared memory
   stat = clinterface_clsetkernelarg_shmem(38, 15, shmem_sub_size) !shared memory
   shmem_sub_size = localworksize_sub(1)*sizeof_integer !hardcode 16 threads
   stat = clinterface_clsetkernelarg_shmem(38, 16, shmem_sub_size) !shared memory

   !clcgstab1D_smy2
   stat = clinterface_clsetkernelarg_ptr(43, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(43, 1, sizeof_ptr, cptr_d_zk)
   stat = clinterface_clsetkernelarg_ptr(43, 2, sizeof_ptr, cptr_d_d)
   stat = clinterface_clsetkernelarg_ptr(43, 3, sizeof_ptr, cptr_d_corder)

   !clcgstab1D_cav
   stat = clinterface_clsetkernelarg_ptr(39, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(39, 1, sizeof_ptr, cptr_d_nij)
   stat = clinterface_clsetkernelarg_ptr(39, 2, sizeof_ptr, cptr_d_nj)
   stat = clinterface_clsetkernelarg_ptr(39, 3, sizeof_ptr, cptr_d_vk)
   stat = clinterface_clsetkernelarg_ptr(39, 4, sizeof_ptr, cptr_d_ap)
   stat = clinterface_clsetkernelarg_ptr(39, 5, sizeof_ptr, cptr_d_zk)
   stat = clinterface_clsetkernelarg_ptr(39, 6, sizeof_ptr, cptr_d_ae)
   stat = clinterface_clsetkernelarg_ptr(39, 7, sizeof_ptr, cptr_d_aw)
   stat = clinterface_clsetkernelarg_ptr(39, 8, sizeof_ptr, cptr_d_an)
   stat = clinterface_clsetkernelarg_ptr(39, 9, sizeof_ptr, cptr_d_as)
   stat = clinterface_clsetkernelarg_ptr(39, 10,sizeof_ptr, cptr_d_at)
   stat = clinterface_clsetkernelarg_ptr(39, 11,sizeof_ptr, cptr_d_ab)
   stat = clinterface_clsetkernelarg_ptr(39, 12,sizeof_ptr, cptr_d_corder)

   !clcgstab1D_caf
   stat = clinterface_clsetkernelarg_ptr(40, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(40, 1, sizeof_ptr, cptr_d_vres)
   stat = clinterface_clsetkernelarg_ptr(40, 2, sizeof_ptr, cptr_d_vk)
   stat = clinterface_clsetkernelarg_ptr(40, 3, sizeof_ptr, cptr_d_res)
   stat = clinterface_clsetkernelarg_ptr(40, 4, sizeof_ptr, cptr_d_vv)
   stat = clinterface_clsetkernelarg_ptr(40, 5, sizeof_ptr, cptr_d_corder)

   !clcgstab1D_cfr
   stat = clinterface_clsetkernelarg_ptr(41, 0, sizeof_ptr, cptr_d_nxyz)
   stat = clinterface_clsetkernelarg_ptr(41, 1, sizeof_ptr, cptr_d_fi)
   stat = clinterface_clsetkernelarg_ptr(41, 2, sizeof_ptr, cptr_d_zk)
   stat = clinterface_clsetkernelarg_ptr(41, 3, sizeof_ptr, cptr_d_res)
   stat = clinterface_clsetkernelarg_ptr(41, 4, sizeof_ptr, cptr_d_reso)
   stat = clinterface_clsetkernelarg_ptr(41, 5, sizeof_ptr, cptr_d_vk)
   stat = clinterface_clsetkernelarg_ptr(41, 6, sizeof_ptr, cptr_d_corder)
   stat = clinterface_clsetkernelarg_ptr(41, 7, sizeof_ptr, cptr_d_alf)
   !call clinterface_clflush(cptr_queue) !flush gpu command queues

end subroutine

subroutine gpu_clinfc_cgstab(NS)

   integer clinterface_clsetkernelarg, clinterface_clenqueuendrangekernel
   external clinterface_clsetkernelarg, clinterface_clenqueuendrangekernel
   integer clinterface_clsetkernelarg_ptr, clinterface_clsetkernelarg_shmem
   external clinterface_clsetkernelarg_ptr, clinterface_clsetkernelarg_shmem

   integer clinterface_dsetmatrix,clinterface_dsetmatrix_nb
   external clinterface_dsetmatrix,clinterface_dsetmatrix_nb
   integer clinterface_dgetmatrix,clinterface_dgetmatrix_nb
   external clinterface_dgetmatrix,clinterface_dgetmatrix_nb

   integer clinterface_igetmatrix,clinterface_igetmatrix_nb
   external clinterface_igetmatrix,clinterface_igetmatrix_nb
   integer clinterface_isetmatrix,clinterface_isetmatrix_nb
   external clinterface_isetmatrix,clinterface_isetmatrix_nb

   integer clinterface_clbarrier, clinterface_clflush
   external clinterface_clbarrier, clinterface_clflush

   integer, intent(in) :: NS

   integer :: bx, by, bz, i, j, k, ijk, n

   integer(kind=4) :: zero = 0, one = 1 !32bit integers
   integer :: nxyz, nxyz_reduce, nxyz_range, L, maxblk
   integer(C_INT) :: stat
   integer(C_SIZE_T) :: shmem_size, shmem_sub_size

   real(kind=8) :: VV0, VRES0, BETOL2
   real(kind=8) :: SMALL = 1.E-30

   integer :: debug_gpu = 1

   nxyz=(nx-1)*(ny-1)*(nz-1) !size of arrays
   nxyz_range=(nx-1)*(ny-1)*(nz-1) !range - certain species only calculate for channel
   nxyz_reduce=(nx-1)*(ny-1)*(nz-1)
   !Globalwork size will change based on species so recalculate
   globalworksize(1) = (((nxyz_range)+(localworksize(1)-1))/localworksize(1))*localworksize(1)
   globalworksize_reduce(1) = (((nxyz_reduce)+(localworksize_reduce(1)-1))/localworksize_reduce(1))*localworksize_reduce(1)

   !this is expensive do we have to do this each time?
   !call gpu_clinterface1D_forder(nx,ny,nz)
   !stat = clinterface_isetmatrix(forder_size,1,FORDER,zero,forder_size,cptr_d_forder,zero,forder_size,cptr_queue)

   !maxblk = (globalworksize(3)-1)*globalworksize(1)*globalworksize(2)+(globalworksize(1)-1)*globalworksize(1)+globalworksize(1)
   maxblk = globalworksize_reduce(1)/threads_reduce-1
   !write(*,*) 'Maximum Reduce Block Size = ',maxblk, nxyz

   !update values in gpu solvers
   stat = clinterface_isetmatrix_nb(1,1,nxyz,zero,1,cptr_d_nxyz,zero,1,cptr_queue) ! Send vector to gpu memory
   stat = clinterface_isetmatrix_nb(1,1,NIJ ,zero,1,cptr_d_nij ,zero,1,cptr_queue) ! Send vector to gpu memory
   stat = clinterface_isetmatrix_nb(1,1,NJ  ,zero,1,cptr_d_nj  ,zero,1,cptr_queue) ! Send vector to gpu memory
   stat = clinterface_isetmatrix_nb(1,1,NJM ,zero,1,cptr_d_njm ,zero,1,cptr_queue) ! Send vector to gpu memory

   ! Kernel Num, Kernel Name
   !          0->?, clcgstab1D_kernel
   !          1->22, clcgstab1D_reduce
   !          2->23, clcgstab1D_reduce_abs
   !          3->27, clcgstab1D_zer
   !          4->28, clcgstab1D_irv
   !          5->29, clcgstab1D_pmd
   !          6->30, clcgstab1D_iwa
   !          7->31, clcgstab1D_cbo
   !          8->32, clcgstab1D_cpk
   !          9->33, clcgstab1D_sfs
   !         10->34, clcgstab1D_sbs
   !         11->35, clcgstab1D_cuk
   !         12->36, clcgstab1D_ruk
   !         13->37, clcgstab1D_ufr
   !         14->38, clcgstab1D_smy
   !         15->39, clcgstab1D_cav
   !         16->40, clcgstab1D_caf
   !         17->41, clcgstab1D_cfr
   !         18->42, clcgstab1D_sfs2
   !         19->43, clcgstab1D_smy2
   !         20->44, clcgstab1D_sbs2
   !         21->45, clcgstab1D_pmd2

   n=1
   CORDER(:)=nxyz
   do k=2,NKM
     do i=2,NIM 
       do j=2,NJM 
          !IJK=LK(K)+LI(I)+J
          ijk=(k-1)*(nx-1)*(ny-1)+(i-1)*(ny-1)+j-1 !minus one to put in c array format
          CORDER(n)=ijk
          n=n+1
        end do
     end do
   end do
   if(pinned .eq. 0) stat = clinterface_isetmatrix(nxyz,1,CORDER,zero,nxyz,cptr_d_corder,zero,nxyz,cptr_queue)

   RSM=1.
   RES0=0.
   RESL=1.
   !clcgstab_zer 27
   stat = clinterface_clenqueuendrangekernel(27,1,globalworksize,localworksize,cptr_queue) !ZERO

   !clcgstab_irv 28
   stat = clinterface_clenqueuendrangekernel(28,1,globalworksize,localworksize,cptr_queue) !CALCULATE INITIAL RESIDUAL VECTOR AND THE NORM

   if(debug_gpu .gt. 5) then
   stat = clinterface_dgetmatrix(nxyz,1,cptr_d_q,zero,nxyz,Q,zero,nxyz,cptr_queue)
   write(*,*) 'max(Q) =',maxval(Q),minval(Q)
   stat = clinterface_dgetmatrix(nxyz,1,cptr_d_fi,zero,nxyz,T,zero,nxyz,cptr_queue)
   write(*,*) 'max(T) =',maxval(T),minval(T)
   stat = clinterface_dgetmatrix(nxyz,1,cptr_d_ap,zero,nxyz,AP,zero,nxyz,cptr_queue)
   write(*,*) 'max(AP) =',maxval(AP),minval(AP)
   stat = clinterface_dgetmatrix(nxyz,1,cptr_d_ae,zero,nxyz,AE,zero,nxyz,cptr_queue)
   write(*,*) 'max(AE) =',maxval(AE),minval(AE)
   stat = clinterface_dgetmatrix(nxyz,1,cptr_d_aw,zero,nxyz,AW,zero,nxyz,cptr_queue)
   write(*,*) 'max(AW) =',maxval(AW),minval(AW)
   stat = clinterface_dgetmatrix(nxyz,1,cptr_d_an,zero,nxyz,AN,zero,nxyz,cptr_queue)
   write(*,*) 'max(AN) =',maxval(AN),minval(AN)
   stat = clinterface_dgetmatrix(nxyz,1,cptr_d_as,zero,nxyz,AS,zero,nxyz,cptr_queue)
   write(*,*) 'max(AS) =',maxval(AS),minval(AS)
   stat = clinterface_dgetmatrix(nxyz,1,cptr_d_at,zero,nxyz,AT,zero,nxyz,cptr_queue)
   write(*,*) 'max(AT) =',maxval(AT),minval(AT)
   stat = clinterface_dgetmatrix(nxyz,1,cptr_d_ab,zero,nxyz,AB,zero,nxyz,cptr_queue)
   write(*,*) 'max(AB) =',maxval(AB),minval(AB)
   stat = clinterface_dgetmatrix(nxyz,1,cptr_d_d,zero,nxyz,D,zero,nxyz,cptr_queue)
   write(*,*) 'max(D) =',maxval(D),minval(D)

   stat = clinterface_dgetmatrix(nxyz,1,cptr_d_res,zero,nxyz,RES,zero,nxyz,cptr_queue)
   write(*,*) 'gpu sum(RES) =',sum(abs(RES)),maxval(RES),minval(RES)

   if(ttime .eq. -1) then
   open (unit=19, file="_T.dat", status='replace', action='write')
      do l=1,(nx-1)*(ny-1)*(nz-1)
          write(19,'(10E18.11,1X)') Q(l),T(l),&
                                     AP(l),AE(l),AW(l),AN(l),&
                                     AS(l),AT(l),AB(l),RES(l)
      end do
   close(19)
   stop
   end if
   ttime=ttime+1
      RES0=0.
      n=1
      do K=2,NKM
        do I=2,NIM 
          do J=2,NJM 
            IJK=LK(K)+LI(I)+J
            RES(IJK)= Q(IJK) - AP(IJK)*T(IJK)-&
     &           AE(IJK)*T(IJK+NJ) - AW(IJK)*T(IJK-NJ) -&
     &           AN(IJK)*T(IJK+1)  - AS(IJK)*T(IJK-1)  -&
     &           AT(IJK)*T(IJK+NIJ)- AB(IJK)*T(IJK-NIJ)
            RES0=RES0+ABS(RES(IJK))
            n=n+1
          end do
        end do
      end do
   write(*,*) 'cpu sum(RES) =',sum(abs(RES)),maxval(RES),minval(RES),ttime,(NKM-1)*(NIM-1)*(NJM-1),n,nxyz
   end if

   !clcreduce_abs
   stat = clinterface_clsetkernelarg_ptr(23, 2 , sizeof_ptr, cptr_d_beto)
   stat = clinterface_clsetkernelarg_ptr(23, 3 , sizeof_ptr, cptr_d_res)
   stat = clinterface_clenqueuendrangekernel(23,1,globalworksize_reduce,localworksize_reduce,cptr_queue) !CALCULATE INITIAL RESIDUAL VECTOR AND THE NORM
   if(pinned .eq. 0) then
     stat = clinterface_dgetmatrix(maxblk,1,cptr_d_beto,zero,maxblk,BETO,zero,maxblk,cptr_queue)
     RES0 = sum(BETO(1:maxblk))
   else
     RES0 = sum(RES)
   end if

   if(debug_gpu .gt. 5) write(*,*) 'RES0 =',RES0,maxval(BETO(1:maxblk)),minval(BETO(1:maxblk))

   !calculate D preconditioner on CPU
   call clcgstab_pmd_cpu
   if(pinned .eq. 0) then 
     stat = clinterface_dsetmatrix(nxyz,1,D,zero,nxyz,cptr_d_d,zero,nxyz,cptr_queue) ! Send vector to gpu memory
     !this will go into the eventlog and execute before the next enqueuekernel
   end if

   !clcgstab_iwa 30
   stat = clinterface_clenqueuendrangekernel(30,1,globalworksize,localworksize,cptr_queue) !CALCULATE ELEMENTS OF PRECONDITIONING MATRIX DIAGONAL
   stat = clinterface_clsetkernelarg_ptr(23, 2 , sizeof_ptr, cptr_d_beto)
   stat = clinterface_clsetkernelarg_ptr(23, 3 , sizeof_ptr, cptr_d_reso)
   stat = clinterface_clenqueuendrangekernel(23,1,globalworksize_reduce,localworksize_reduce,cptr_queue) !CALCULATE INITIAL RESIDUAL VECTOR AND THE NORM
   if(pinned .eq. 0) then 
     stat = clinterface_dgetmatrix(maxblk,1,cptr_d_beto,zero,maxblk,BETO,zero,maxblk,cptr_queue)
     RES0 = sum(BETO(1:maxblk))
   else
     RES0 = sum(RES)
   endif

   OM=1.
   ALF=1.
   GAM=1.
   BETO(:)=0.
   BETOL=1.
   BETOL2=1.
   VRES0=0.
   VV0=1.
   L=1

   if(debug_gpu .gt. 5) write(*,*) 'RESMAX =',RESMAX
   do while(RSM .gt. RESMAX .and. L .lt. NS)
     L=L+1

     !clcgstab_cbo
     stat = clinterface_clenqueuendrangekernel(31,1,globalworksize,localworksize,cptr_queue) !CALCULATE BETA AND OMEGA
     
     !clcgstab_reduce
     stat = clinterface_clsetkernelarg_ptr(22, 2 , sizeof_ptr, cptr_d_beto)
     stat = clinterface_clsetkernelarg_ptr(22, 3 , sizeof_ptr, cptr_d_bet)
     stat = clinterface_clenqueuendrangekernel(22,1,globalworksize_reduce,localworksize_reduce,cptr_queue) !CALCULATE INITIAL RESIDUAL VECTOR AND THE NORM
     if(pinned .eq. 0) then
       stat = clinterface_dgetmatrix(maxblk,1,cptr_d_beto,zero,maxblk,BETO,zero,maxblk,cptr_queue)
       BETOL2 = sum(BETO(1:maxblk))
     else
       BETOL2 = sum(BET)
     end if
     OM = BETOL2*GAM/(ALF*BETOL+SMALL)
     BETOL = BETOL2
     if(debug_gpu .gt. 5) write(*,*) 'BETO =',BETOL,OM,SMALL

     !clcgstab_cpk
     !stat = clinterface_clsetkernelarg(32, 5, sizeof_double, OM)
     stat = clinterface_dsetmatrix(1,1,OM,zero,1,cptr_d_om,zero,1,cptr_queue) ! Send vector to gpu memory
     !stat = clinterface_clsetkernelarg(32, 6, sizeof_double, ALF)
     stat = clinterface_dsetmatrix(1,1,ALF,zero,1,cptr_d_alf,zero,1,cptr_queue) ! Send vector to gpu memory
     stat = clinterface_clenqueuendrangekernel(32,1,globalworksize,localworksize,cptr_queue) !CALCULATE PK
     if(debug_gpu .gt. 5) write(*,*) 'OM,ALF =',OM,ALF

     !clcgstab1D_sfs_init
     stat = clinterface_clenqueuendrangekernel(46,1,globalworksize_sub,localworksize_sub,cptr_queue) !INIT FORWARD SUBSTITUTION

     !clcgstab_sfs
     stat = clinterface_clenqueuendrangekernel(33,1,globalworksize_sub,localworksize_sub,cptr_queue) !SOLVE (M ZK = PK) - FORWARD SUBSTITUTION

     !clcgstab_sfs2
     stat = clinterface_clenqueuendrangekernel(42,1,globalworksize,localworksize,cptr_queue) ! FOWARD SUBSTITUTION PART 2

     !clcgstab1D_sfs_init
     stat = clinterface_clenqueuendrangekernel(46,1,globalworksize_sub,localworksize_sub,cptr_queue) !INIT BACKWARD SUBSTITUTION

     !clcgstab_sbs
     stat = clinterface_clenqueuendrangekernel(34,1,globalworksize_sub,localworksize_sub,cptr_queue) !SOLVE BACKWARD SUBSTITUTION

     !clcgstab_sbs2
     !stat = clinterface_clenqueuendrangekernel(44,1,globalworksize_sub,localworksize,cptr_queue) !SOLVE BACKWARD SUBSTITUTION PART2

     !clcgstab_cuk
     stat = clinterface_clenqueuendrangekernel(35,1,globalworksize,localworksize,cptr_queue) !CALCULATE UK = A.PK

     !clcgstab_ruk
     stat = clinterface_clenqueuendrangekernel(36,1,globalworksize,localworksize,cptr_queue) !CALCULATE SCALAR PRODUCT UK.RESO AND GAMMA

     !clcgstab_reduce
     stat = clinterface_clsetkernelarg_ptr(22, 2 , sizeof_ptr, cptr_d_beto)
     stat = clinterface_clsetkernelarg_ptr(22, 3 , sizeof_ptr, cptr_d_vv)
     stat = clinterface_clenqueuendrangekernel(22,1,globalworksize_reduce,localworksize_reduce,cptr_queue) !CALCULATE INITIAL RESIDUAL VECTOR AND THE NORM
     if(pinned .eq. 0) then
       stat = clinterface_dgetmatrix(maxblk,1,cptr_d_beto,zero,maxblk,BETO,zero,maxblk,cptr_queue)
       UKRESO = sum(BETO(1:maxblk))
     else
       UKRESO = sum(VV)
     endif
     GAM = BETOL/(UKRESO+SMALL)
     if(debug_gpu .gt. 5) write(*,*) 'GAM =',GAM,UKRESO

     !clcgstab_ufr
     !stat = clinterface_clsetkernelarg(37, 6, sizeof_double, GAM)
     stat = clinterface_dsetmatrix(1,1,GAM,zero,1,cptr_d_gam,zero,1,cptr_queue) ! Send vector to gpu memory
     stat = clinterface_clenqueuendrangekernel(37,1,globalworksize,localworksize,cptr_queue) !UPDATE (FI) AND CALCULATE W (OVERwrite RES - IT IS RES-UPDATE)

     if(debug_gpu .gt. 5) then
       stat = clinterface_dgetmatrix(nxyz,1,cptr_d_fi,zero,nxyz,T,zero,nxyz,cptr_queue)
       write(*,*) 'max(FI) =',maxval(T),minval(T)
       stat = clinterface_dgetmatrix(nxyz,1,cptr_d_res,zero,nxyz,RES,zero,nxyz,cptr_queue)
       write(*,*) 'max(RES) =',maxval(RES),minval(RES),SUM(RES)
       ZK=0
       stat = clinterface_dgetmatrix(nxyz,1,cptr_d_zk,zero,nxyz,ZK,zero,nxyz,cptr_queue)
       write(*,*) 'BBefore max(ZK) =',maxval(ZK),minval(ZK)
     end if

     !clcgstab1D_sfs_init
     stat = clinterface_clenqueuendrangekernel(46,1,globalworksize_sub,localworksize_sub,cptr_queue) !INIT FORWARD SUBSTITUTION

     !clcgstab_smy
     stat = clinterface_clenqueuendrangekernel(38,1,globalworksize_sub,localworksize_sub,cptr_queue) !SOLVE (M Y = W); Y OVERwriteS ZK; FORWARD SUBSTITUTION

     if(debug_gpu .gt. 5) then
       stat = clinterface_dgetmatrix(nxyz,1,cptr_d_as,zero,nxyz,AS,zero,nxyz,cptr_queue)
       write(*,*) 'max(AS) =',maxval(AS),minval(AS),sum(AS)
       stat = clinterface_dgetmatrix(nxyz,1,cptr_d_aw,zero,nxyz,AW,zero,nxyz,cptr_queue)
       write(*,*) 'max(AW) =',maxval(AW),minval(AW),sum(AW)
       stat = clinterface_dgetmatrix(nxyz,1,cptr_d_ab,zero,nxyz,AB,zero,nxyz,cptr_queue)
       write(*,*) 'max(AB) =',maxval(AB),minval(AB),sum(AB)

       stat = clinterface_dgetmatrix(nxyz,1,cptr_d_d,zero,nxyz,D,zero,nxyz,cptr_queue)
       write(*,*) 'max(D) =',maxval(D),minval(D),sum(D)
       ZK=0
       stat = clinterface_dgetmatrix(nxyz,1,cptr_d_zk,zero,nxyz,ZK,zero,nxyz,cptr_queue)
       write(*,*) 'Before max(ZK) =',maxval(ZK),minval(ZK)
     end if

     !clcgstab_smy2
     stat = clinterface_clenqueuendrangekernel(43,1,globalworksize,localworksize_sub,cptr_queue) !FORWARD SUBSTITUTION PART 2

     !clcgstab1D_sfs_init
     stat = clinterface_clenqueuendrangekernel(46,1,globalworksize_sub,localworksize_sub,cptr_queue) !INIT BACKWARD SUBSTITUTION

     !clcgstab_sbs
     stat = clinterface_clenqueuendrangekernel(34,1,globalworksize_sub,localworksize_sub,cptr_queue) !SOLVE BACKWARD SUBSTITUTION


     !clcgstab_sbs2
     !stat = clinterface_clenqueuendrangekernel(44,1,globalworksize,localworksize,cptr_queue) !SOLVE BACKWARD SUBSTITUTION PART2

     !clcgstab_cav
     stat = clinterface_clenqueuendrangekernel(39,1,globalworksize,localworksize,cptr_queue) !CALCULATE V = A.Y (VK = A.ZK)

     !clcgstab_caf
     stat = clinterface_clenqueuendrangekernel(40,1,globalworksize,localworksize,cptr_queue) !CALCULATE ALPHA (ALF)

     !clcgstab_reduce
     stat = clinterface_clsetkernelarg_ptr(22, 2 , sizeof_ptr, cptr_d_beto)
     stat = clinterface_clsetkernelarg_ptr(22, 3 , sizeof_ptr, cptr_d_vres)
     stat = clinterface_clenqueuendrangekernel(22,1,globalworksize_reduce,localworksize_reduce,cptr_queue) !CALCULATE INITIAL RESIDUAL VECTOR AND THE NORMi
     if(pinned .eq. 0) then 
       stat = clinterface_dgetmatrix(maxblk,1,cptr_d_beto,zero,maxblk,BETO,zero,maxblk,cptr_queue)
       VRES0 = sum(BETO(1:maxblk))
     else
       VRES0 = sum(VRES)
     endif

     !clcgstab_reduce
     stat = clinterface_clsetkernelarg_ptr(22, 2 , sizeof_ptr, cptr_d_beto)
     stat = clinterface_clsetkernelarg_ptr(22, 3 , sizeof_ptr, cptr_d_vv)
     stat = clinterface_clenqueuendrangekernel(22,1,globalworksize_reduce,localworksize_reduce,cptr_queue) !CALCULATE INITIAL RESIDUAL VECTOR AND THE NORMi
     if(pinned .eq. 0) then
       stat = clinterface_dgetmatrix(maxblk,1,cptr_d_beto,zero,maxblk,BETO,zero,maxblk,cptr_queue)
       VV0 = sum(BETO(1:maxblk))
     else
       VV0 = sum(VV)
     endif
     ALF=VRES0/(VV0+SMALL);

     if(debug_gpu .gt. 5) write(*,*) 'VRES0,VV0,ALF =',VRES0,VV0,ALF

     !clcgstab_cfr
     !stat = clinterface_clsetkernelarg(41, 7, sizeof_double, ALF)
     stat = clinterface_dsetmatrix(1,1,ALF,zero,1,cptr_d_alf,zero,1,cptr_queue) ! Send vector to gpu memory
     stat = clinterface_clenqueuendrangekernel(41,1,globalworksize,localworksize,cptr_queue) !UPDATE VARIABLE (FI) AND RESIDUAL (RES) VECTORS

     !clcgstab_reduce_abs
     stat = clinterface_clsetkernelarg_ptr(23, 2 , sizeof_ptr, cptr_d_beto)
     stat = clinterface_clsetkernelarg_ptr(23, 3 , sizeof_ptr, cptr_d_res)
     stat = clinterface_clenqueuendrangekernel(23,1,globalworksize_reduce,localworksize_reduce,cptr_queue) !CALCULATE INITIAL RESIDUAL VECTOR AND THE NORM
     if(pinned .eq. 0) then 
       stat = clinterface_dgetmatrix(maxblk,1,cptr_d_beto,zero,maxblk,BETO,zero,maxblk,cptr_queue)
       RESL = sum(BETO(1:maxblk))
       write(*,*) 'RESL =',RESL
     else
       RESL = sum(RES)
     endif
     RSM = RESL/(RES0+SMALL) !this comparison of the orginal residual(RESO) to the new residual can run into trouble, especially for hydrogen.
     if(L .lt. 2) RSM = 1 !force atleast two iterations

     write(*,*) L,' SWEEP, RESL = ',RESL,' RSM = ',RSM,' RES0 = ',RES0

     !My understanding if the RESL value is like a NORM except without the sqrt.
     !If this is the case we should be able to to just take the NORM and not have to divide by the orginal NORM
     !the orginial NORM that was taken at the beginning.

     !RSM = RESL !setting the convergence threshold equal to the NORM not NORM/NORM0

   end do

end subroutine

subroutine clcgstab_pmd_cpu

  integer :: I,J,K,IJK
!.....CALCULATE ELEMENTS OF PRECONDITIONING MATRIX DIAGONAL

!
      do K=2,NKM
        do I=2,NIM
          do J=2,NJM
            IJK=LK(K)+LI(I)+J
            D(IJK)=1./(AP(IJK) - AW(IJK)*D(IJK-NJ)*AE(IJK-NJ) -&
     &             AS(IJK)*D(IJK-1)*AN(IJK-1) -&
     &             AB(IJK)*D(IJK-NIJ)*AT(IJK-NIJ))
          end do
        end do
      end do

end subroutine
end module
