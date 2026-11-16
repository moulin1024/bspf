// Native mapped FD4, right-preconditioned BiCGSTAB, FFTW DST Poisson inverse.
#include "../poisson_campaign/native_common.hpp"
#include <fftw3.h>
using Vec=std::vector<double>;
struct Solver {
 int n,m; double h; Vec A,B,C,G,H,tmp,full,work,inv; fftw_plan plan;
 struct Stencil {int start,count; std::array<double,6>w{};};
 std::vector<Stencil>d1,d2;
 Solver(const Bundle&p):n(p.at("A").r),m(n-2),h(1./(n-1)),A(p.at("A").v),B(p.at("B").v),C(p.at("C").v),G(p.at("G").v),H(p.at("H").v),tmp(n*n),full(n*n),work(m*m),inv(m*m),d1(n),d2(n){
 if(n<7)throw std::runtime_error("Need n>=7");
 for(int i=0;i<n;++i){
  int k=std::min(i,n-1-i); bool reflect=i>n/2;
  Stencil s,t; s.count=5;t.count=5;
  if(k==0){s.start=0;s.w={-25./12,4,-3,4./3,-.25,0};}
  else if(k==1){s.start=0;s.w={-.25,-5./6,1.5,-.5,1./12,0};}
  else{s.start=k-2;s.w={1./12,-2./3,0,2./3,-1./12,0};}
  if(k<=1){t.start=0;t.count=6;t.w={10./12,-15./12,-4./12,14./12,-6./12,1./12};}
  else{t.start=k-2;t.w={-1./12,4./3,-2.5,4./3,-1./12,0};}
  if(reflect){s.start=n-s.start-s.count;t.start=n-t.start-t.count;std::reverse(s.w.begin(),s.w.begin()+s.count);std::reverse(t.w.begin(),t.w.begin()+t.count);for(double&v:s.w)v=-v;}
  for(double&v:s.w)v/=h;for(double&v:t.w)v/=h*h;d1[i]=s;d2[i]=t;
 }
 double a=0,c=0;for(int i=1;i<n-1;++i)for(int j=1;j<n-1;++j){a+=A[i*n+j];c+=C[i*n+j];}a/=m*m;c/=m*m;
 const double pi=std::acos(-1.);
 for(int i=0;i<m;++i)for(int j=0;j<m;++j)inv[i*m+j]=1./(4/(h*h)*(a*std::pow(std::sin(pi*(i+1)/(2*(n-1))),2)+c*std::pow(std::sin(pi*(j+1)/(2*(n-1))),2)))*1./(4*(n-1)*(n-1));
 plan=fftw_plan_r2r_2d(m,m,work.data(),work.data(),FFTW_RODFT00,FFTW_RODFT00,FFTW_ESTIMATE);if(!plan)throw std::runtime_error("FFTW planning failed");
 }
 ~Solver(){fftw_destroy_plan(plan);}
 void apply_full(const Vec&u,Vec&out){
 for(int i=0;i<n;++i)for(int j=0;j<n;++j){auto&s=d1[j];double v=0;for(int k=0;k<s.count;++k)v+=s.w[k]*u[i*n+s.start+k];tmp[i*n+j]=v;}
 for(int i=1;i<n-1;++i)for(int j=1;j<n-1;++j){double ds=0,dss=0,dtt=0,dst=0;auto&s=d1[i];auto&ss=d2[i];auto&tt=d2[j];for(int k=0;k<s.count;++k){ds+=s.w[k]*u[(s.start+k)*n+j];dst+=s.w[k]*tmp[(s.start+k)*n+j];}for(int k=0;k<ss.count;++k)dss+=ss.w[k]*u[(ss.start+k)*n+j];for(int k=0;k<tt.count;++k)dtt+=tt.w[k]*u[i*n+tt.start+k];int z=i*n+j;out[(i-1)*m+j-1]=-(A[z]*dss+2*B[z]*dst+C[z]*dtt+G[z]*ds+H[z]*tmp[z]);}
 }
 void apply(const Vec&u,Vec&out){std::fill(full.begin(),full.end(),0);for(int i=0;i<m;++i)std::copy_n(u.data()+i*m,m,full.data()+(i+1)*n+1);apply_full(full,out);}
 void prec(const Vec&u,Vec&out){std::copy(u.begin(),u.end(),work.begin());fftw_execute(plan);for(size_t k=0;k<work.size();++k)work[k]*=inv[k];fftw_execute(plan);std::copy(work.begin(),work.end(),out.begin());}
};
double dot(const Vec&a,const Vec&b){double s=0;for(size_t k=0;k<a.size();++k)s+=a[k]*b[k];return s;}
double norm(const Vec&a){return std::sqrt(dot(a,a));}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("usage: fd4 input.bin output_prefix");auto input=load(argv[1]);auto setup=Clock::now();Solver op(input);double setup_seconds=elapsed(setup);int N=op.m*op.m;Vec x(N),b(N),r(N),shadow(N),p(N),v(N),s(N),t(N),ph(N),sh(N),ax(N);const auto&forcing=input.at("forcing").v;auto start=Clock::now();op.apply_full(input.at("boundary").v,b);for(int i=0;i<op.m;++i)for(int j=0;j<op.m;++j){int k=i*op.m+j;b[k]=forcing[(i+1)*op.n+j+1]-b[k];}r=b;shadow=r;double bn=norm(b),rho0=1,alpha=1,omega=1;int iterations=0;bool converged=false;
 for(int it=1;it<=1000;++it){iterations=it;double rho=dot(shadow,r);if(!std::isfinite(rho)||std::abs(rho)<1e-290)break;double beta=rho/rho0*alpha/omega;for(int k=0;k<N;++k)p[k]=r[k]+beta*(p[k]-omega*v[k]);op.prec(p,ph);op.apply(ph,v);double den=dot(shadow,v);if(std::abs(den)<1e-290)break;alpha=rho/den;for(int k=0;k<N;++k)s[k]=r[k]-alpha*v[k];
 if(norm(s)<=1e-12*bn){for(int k=0;k<N;++k)x[k]+=alpha*ph[k];op.apply(x,ax);for(int k=0;k<N;++k)r[k]=b[k]-ax[k];if(norm(r)<=2e-12*bn){converged=true;break;}shadow=r;std::fill(p.begin(),p.end(),0);std::fill(v.begin(),v.end(),0);rho0=alpha=omega=1;continue;}
 op.prec(s,sh);op.apply(sh,t);double tt=dot(t,t);if(tt<1e-290)break;omega=dot(t,s)/tt;if(std::abs(omega)<1e-290)break;for(int k=0;k<N;++k){x[k]+=alpha*ph[k]+omega*sh[k];r[k]=s[k]-omega*t[k];}rho0=rho;
 if(norm(r)<=1e-12*bn){op.apply(x,ax);for(int k=0;k<N;++k)r[k]=b[k]-ax[k];if(norm(r)<=2e-12*bn){converged=true;break;}shadow=r;std::fill(p.begin(),p.end(),0);std::fill(v.begin(),v.end(),0);rho0=alpha=omega=1;}
 }
 op.apply(x,ax);for(int k=0;k<N;++k)r[k]=b[k]-ax[k];double residual=norm(r)/bn;Mat solution=input.at("boundary");for(int i=0;i<op.m;++i)for(int j=0;j<op.m;++j)solution.v[(i+1)*op.n+j+1]+=x[i*op.m+j];double seconds=elapsed(start);save(std::string(argv[2])+".bin",{{"solution",solution}});std::ofstream f(std::string(argv[2])+".json");f<<std::setprecision(17)<<"{\"solve_seconds\":"<<seconds<<",\"setup_seconds\":"<<setup_seconds<<",\"iterations\":"<<iterations<<",\"relative_residual\":"<<residual<<",\"converged\":"<<(converged?"true":"false")<<",\"solve_calls\":1,\"warmups\":0}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
