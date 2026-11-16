#pragma once
#include "../poisson_campaign/native_common.hpp"
struct Poisson {
  Bundle p; Blas blas; size_t n; double residual=0,preconditioned_defect=0;
  Mat modal_b,modal_d; int iterations=0,refinements=0;std::vector<double> defect_history;
  Poisson(const std::string& input):p(load(input)),blas("/System/Library/Frameworks/Accelerate.framework/Accelerate"),n(p.at("B").c){
    Mat bi(p.at("B").r,n-2),di(bi.r,n-2);
    for(size_t i=0;i<bi.r;++i)for(size_t j=0;j<n-2;++j){bi.v[i*(n-2)+j]=p.at("B").v[i*n+j+1];di.v[i*(n-2)+j]=p.at("D").v[i*n+j+1];}
    modal_b=blas.mm(bi,p.at("V"));modal_d=blas.mm(di,p.at("V"));
  }
  Mat interior(const Mat&a){Mat b(n-2,n-2);for(size_t i=1;i<n-1;++i)for(size_t j=1;j<n-1;++j)b.v[(i-1)*(n-2)+j-1]=a.v[i*n+j];return b;}
  Mat extend(const Mat&a){Mat b(n,n);for(size_t i=1;i<n-1;++i)for(size_t j=1;j<n-1;++j)b.v[i*n+j]=a.v[(i-1)*(n-2)+j-1];return b;}
  Mat full(const Mat&a){
    const Mat &b=p.at("B"),&d=p.at("D");
    Mat us=blas.mm(blas.mm(d,a),b,false,true),ut=blas.mm(blas.mm(b,a),d,false,true),fs=us,ft=ut;
    const auto &ss=p.at("ss").v,&st=p.at("st").v,&tt=p.at("tt").v;
    for(size_t i=0;i<us.size();++i){fs.v[i]=ss[i]*us.v[i]+st[i]*ut.v[i];ft.v[i]=st[i]*us.v[i]+tt[i]*ut.v[i];}
    Mat y=blas.mm(blas.mm(d,fs,true),b);axpy(y,blas.mm(blas.mm(b,ft,true),d));return y;
  }
  Mat fast_full(const Mat&x){
    Mat y(n,n);int terms=int(p.at("terms").v[0]);
    if(!terms)return full(x);
    for(int i=0;i<terms;++i)axpy(y,blas.mm(blas.mm(p.at("left"+std::to_string(i)),x),p.at("right"+std::to_string(i))));
    return y;
  }
  // Interior CG in the generalized eigenbasis: diagonal preconditioning.
  Mat apply(const Mat&a){
    const Mat &b=modal_b,&d=modal_d;
    Mat us=blas.mm(blas.mm(d,a),b,false,true),ut=blas.mm(blas.mm(b,a),d,false,true),fs=us,ft=ut;
    const auto &ss=p.at("ss").v,&st=p.at("st").v,&tt=p.at("tt").v;
    for(size_t i=0;i<us.size();++i){fs.v[i]=ss[i]*us.v[i]+st[i]*ut.v[i];ft.v[i]=st[i]*us.v[i]+tt[i]*ut.v[i];}
    Mat y=blas.mm(blas.mm(d,fs,true),b);axpy(y,blas.mm(blas.mm(b,ft,true),d));return y;
  }
  Mat diagonal(const Mat&a){Mat z=a;const auto& inv=p.at("inverse").v;for(size_t i=0;i<z.size();++i)z.v[i]*=inv[i];return z;}
  Mat prec(const Mat&a){const Mat&v=p.at("V");Mat z=blas.mm(blas.mm(v,a,true),v);for(size_t i=0;i<z.size();++i)z.v[i]*=p.at("inverse").v[i];return blas.mm(blas.mm(v,z),v,false,true);}
  Mat pcg(const Mat&nodal_rhs,double tolerance){
    const Mat&v=p.at("V");Mat rhs=blas.mm(blas.mm(v,nodal_rhs,true),v);
    double target=std::max(tolerance*blas.norm(rhs),1e-28);
    Mat x(n-2,n-2),r=rhs,z=diagonal(r),d=z;double rho=blas.dot(r,z);
    while(blas.norm(r)>target&&iterations<1000){
      Mat ad=apply(d);double pap=blas.dot(d,ad);if(!(pap>0&&rho>0))throw std::runtime_error("PCG breakdown");
      axpy(x,d,rho/pap);axpy(r,ad,-rho/pap);++iterations;
      bool refresh=iterations%40==0||blas.norm(r)<=target;
      if(refresh)r=added(rhs,apply(x),-1);
      z=diagonal(r);double next=blas.dot(r,z);
      if(refresh)d=z;else for(size_t i=0;i<d.size();++i)d.v[i]=z.v[i]+next/rho*d.v[i];rho=next;
    }
    if(iterations>=1000)throw std::runtime_error("PCG iteration limit");
    return blas.mm(blas.mm(v,x),v,false,true);
  }
  Mat solve(){
    iterations=0;refinements=0;defect_history.clear();
    const Mat&b=p.at("B");
    // New campaign inputs keep all RHS-dependent weighting inside solve time.
    // Retain the legacy weighted bundle format for archived harnesses.
    Mat weighted;
    if(p.count("forcing")||p.count("forcing_nodes")){
      // RHS-dependent transfer stays inside the timed solve.
      if(p.count("forcing_nodes")){
        const Mat&fn=p.at("forcing_nodes");if(fn.r!=n||fn.c!=n)throw std::runtime_error("Invalid nodal forcing shape");
        weighted=blas.mm(blas.mm(b,fn),b,false,true);
      }else weighted=p.at("forcing");
      const Mat&w=p.at("weights");same(weighted,w);
      for(size_t i=0;i<weighted.size();++i)weighted.v[i]*=w.v[i];
    }else weighted=p.at("weighted_forcing");
    Mat load=blas.mm(blas.mm(b,weighted,true),b);
    Mat g=p.at("boundary");
    // Use one operator for BOTH boundary lifting and interior iteration.
    Mat rhs=interior(added(load,fast_full(g),-1));
    double norm=blas.norm(interior(added(load,full(g),-1)));
    axpy(g,extend(pcg(rhs,1e-13)));
    // Compression error and cancellation can be magnified at high resolution.
    // Correct against the ORIGINAL quadrature operator; the exact MMS is never
    // used. The preconditioned defect is a forward-sensitive stopping measure.
    Mat r=interior(added(load,full(g),-1));
    double scale=std::max(blas.norm(g),1e-300);
    preconditioned_defect=blas.norm(prec(r))/scale;
    defect_history.push_back(preconditioned_defect);
    while(preconditioned_defect>1e-13&&refinements<3){
      Mat candidate=g;axpy(candidate,extend(pcg(r,1e-2)));
      Mat next=interior(added(load,full(candidate),-1));
      double indicator=blas.norm(prec(next))/std::max(blas.norm(candidate),1e-300);
      if(!std::isfinite(indicator)||indicator>=preconditioned_defect)break;
      g=std::move(candidate);r=std::move(next);preconditioned_defect=indicator;
      defect_history.push_back(indicator);++refinements;
    }
    residual=blas.norm(r)/std::max(norm,1e-300);
    if(!finite(g)||!std::isfinite(residual)||residual>2e-11||preconditioned_defect>1e-12||iterations>=1000)
      throw std::runtime_error("Original quadrature residual/defect check failed");
    return g;
  }
};
