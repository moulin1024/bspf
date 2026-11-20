subroutine init_mms(q,w,k) bind(C)
  use gauss
  use mms_profile
  use basis_at_gaussian
  use mod_basisfunctions
  use iso_c_binding
  implicit none
  real(c_double) :: q(n_gauss),w(n_gauss)
  real(c_double), value :: k
  integer :: i,j
  wave=k
  wgauss=w
  do j=1,n_gauss
    do i=1,n_gauss
      call basisfunctions(q(i),q(j),h(:,:,i,j),h_s(:,:,i,j),h_t(:,:,i,j))
    end do
  end do
end subroutine

subroutine mms_element(x,size,mat,rhs) bind(C)
  use iso_c_binding
  use data_structure
  use mms_profile, only: tilted
  implicit none
  real(c_double) :: x(2,4,4),size(4,4),mat(16,16),rhs(16)
  type(type_node) :: nodes(4)
  type(type_element) :: elem
  integer :: i
  do i=1,4
    nodes(i)%x(1,:,:)=transpose(x(:,:,i))
    nodes(i)%values=0
    if (tilted) nodes(i)%values(1,:,1)=x(1,:,i)
  end do
  elem%size=transpose(size)
  call element_matrix_GS(.false.,0,[0d0,0d0],0d0,1d0,elem,nodes,1,1,1,mat,rhs)
end subroutine


subroutine set_tilted_mms(enabled) bind(C)
  use iso_c_binding
  use mms_profile
  integer(c_int), value :: enabled
  tilted = enabled /= 0
end subroutine
