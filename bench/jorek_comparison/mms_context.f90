! Small axisymmetric driver context for JOREK's unmodified element_matrix_GS.
module mod_parameters
  implicit none
  integer, parameter :: n_order=3, n_degrees=4, n_vertex_max=4
  logical, parameter :: with_TiTe=.false.
end module
module data_structure
  type type_element
    real*8 :: size(4,4)
  end type
  type type_node
    real*8 :: x(1,4,2), values(1,4,1)
  end type
end module
module gauss
  integer, parameter :: n_gauss=MMS_GAUSS_ORDER
  real*8 :: wgauss(n_gauss)
end module
module basis_at_gaussian
  use gauss, only: n_gauss
  real*8 :: h(4,4,n_gauss,n_gauss), h_s(4,4,n_gauss,n_gauss), h_t(4,4,n_gauss,n_gauss)
end module
module mms_profile
  logical :: tilted = .false.
  real*8 :: wave
end module

subroutine density(xp,xc,z,zx,psi,pa,pb,n,np,nz,npp,nzz,npz,nppp,nzzp,nppz)
  logical :: xp
  integer :: xc
  real*8 :: z,zx(2),psi,pa,pb,n,np,nz,npp,nzz,npz,nppp,nzzp,nppz
  n=1; np=0; nz=0; npp=0; nzz=0; npz=0; nppp=0; nzzp=0; nppz=0
end subroutine

subroutine temperature(xp,xc,z,zx,psi,pa,pb,t,tp,tz,tpp,tzz,tpz,tppp,tzzp,tppz)
  use mms_profile
  logical :: xp
  integer :: xc
  real*8 :: z,zx(2),psi,pa,pb,t,tp,tz,tpp,tzz,tpz,tppp,tzzp,tppz
  tp=wave**2*cos(wave*z)
  t=psi*tp; tz=-psi*wave**3*sin(wave*z)
  tpp=0; tzz=-psi*wave**2*tp; tpz=-wave**3*sin(wave*z)
  tppp=0; tzzp=-wave**2*tp; tppz=0
end subroutine

subroutine FFprime(xp,xc,z,zx,psi,pa,pb,f,fp,fz,fpp,fzz,fpz,flag)
  use mms_profile
  logical :: xp,flag
  integer :: xc
  real*8 :: z,zx(2),psi,pa,pb,f,fp,fz,fpp,fzz,fpz
  f=-wave**2*cos(wave*z); fp=0; fz=wave**3*sin(wave*z)
  fpp=0; fzz=-wave**2*f; fpz=0
  if (tilted) then
    ! For the absolute linear GS solve the driver passes R as the profile argument.
    ! This adds Delta* of a Gaussian; the existing temperature term supplies the background.
    block
      real*8 :: r,u,v,ur,uz,vr,vz,g,lr,lz
      r=psi
      ur=0.8d0/0.08d0; uz=0.6d0/0.08d0
      vr=-0.6d0/0.24d0; vz=0.8d0/0.24d0
      u=ur*(r-1.30d0)+uz*(z-0.72d0)
      v=vr*(r-1.30d0)+vz*(z-0.72d0)
      g=0.6d0*exp(-0.5d0*(u*u+v*v))
      lr=-u*ur-v*vr; lz=-u*uz-v*vz
      f=f+g*(lr*lr+lz*lz-ur*ur-uz*uz-vr*vr-vz*vz-lr/r)
    end block
  end if
end subroutine
