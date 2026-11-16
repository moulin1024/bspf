"""Additional independent checks; requires the benchmark's CUDA environment."""
import json
import numpy as np
import torch
from run import DATA, ROOT, Sampler, gpu, reference_grid
from mms import CurvedBox, PlaneWaveMMS
from solver import FCOperator, fgmres

@torch.inference_mode()
def main():
    torch.backends.cuda.matmul.allow_tf32=False
    rng=np.random.default_rng(189);X=rng.uniform(.05,.95,(57,3))
    original=PlaneWaveMMS.load(DATA).evaluate(X)
    u,f=Sampler('turbulence')(X)
    errors={}
    for name,actual,ref in [('sampler_value',u,original['velocity'][:,0]),
                           ('sampler_forcing',f,original['vector_poisson_forcing'][:,0])]:
        errors[name]=float(np.linalg.norm(actual.cpu().numpy()-ref)/np.linalg.norm(ref))
        assert errors[name]<1e-12
    n=17;m=n-2;metric=np.zeros((m,m,m,6));metric[...,:3]=1
    A=FCOperator(n,metric,np.zeros((m,m,m,3)))
    zero,info=fgmres(A,torch.zeros(m**3,device='cuda',dtype=torch.float64))
    assert info['iterations']==0 and zero.count_nonzero().item()==0
    errors['zero_rhs_iterations']=0
    x=gpu(rng.normal(size=m**3));rhs=A(x)
    solved,info=fgmres(A,rhs,rtol=1e-10)
    errors['reference_box_solve_relative_error']=(torch.linalg.vector_norm(solved-x)/torch.linalg.vector_norm(x)).item()
    errors['reference_box_iterations']=info['iterations']
    assert errors['reference_box_solve_relative_error']<1e-8
    # Independently form the FC Fourier evaluation at the original nodes.
    E=A.E.cpu().numpy();freq=A.freq.cpu().numpy();t=np.linspace(0,1,n)
    B=(np.exp(1j*(t[:,None]+25/(n-1))*freq[None,:])@np.fft.fft(E,axis=0)/len(E)).real
    errors['interpolant_nodal_identity_error']=float(np.max(abs(B-np.eye(n))))
    assert errors['interpolant_nodal_identity_error']<1e-9
    out=ROOT/'evidence/fc_pytorch_rtx3070/validation.json'
    out.write_text(json.dumps(errors,indent=2)+'\n');print(json.dumps(errors,indent=2))
if __name__=='__main__':main()
