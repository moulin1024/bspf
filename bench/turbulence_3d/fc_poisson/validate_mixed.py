"""Independent Torch-autodiff check and mixed scalar sampler cross-check on CUDA."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from run import Sampler,ROOT
from mms import PlaneWaveMMS
from poisson_features import PoissonFeatures


def main():
    p=argparse.ArgumentParser();p.add_argument('--modes',type=Path,default=ROOT/'evidence/nonperiodic_curved_mixed/modes.npz');args=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    cpu=PoissonFeatures.load(args.modes);gpu=PoissonFeatures.load(args.modes,xp=torch,device='cuda')
    rng=np.random.default_rng(75);direction=rng.normal(size=(48,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
    pts=np.r_[rng.random((32,3)),cpu.gaussian_centers,cpu.tanh_centers,
        cpu.tanh_centers[0]+direction*(cpu.tanh_radii[0]+rng.uniform(-.025,.025,(48,1)))]
    x=torch.tensor(pts,dtype=torch.float64,device='cuda',requires_grad=True)
    value,forcing=gpu.evaluate(x)
    grad=torch.autograd.grad(value.sum(),x,create_graph=True)[0]
    lap=sum(torch.autograd.grad(grad[:,i].sum(),x,retain_graph=True)[0][:,i] for i in range(3))
    ad_error=(torch.linalg.vector_norm(lap+forcing)/torch.linalg.vector_norm(forcing)).item()
    assert ad_error<1e-12
    with torch.inference_mode():
        exact,f=Sampler('turbulence',args.modes)(pts)
        old=PlaneWaveMMS.load(args.modes).evaluate(pts);extra,ef=cpu.evaluate(pts)
        refs=[old['velocity'][:,0]+extra,old['vector_poisson_forcing'][:,0]+ef]
        errors=[float(np.linalg.norm(v.cpu().numpy()-r)/np.linalg.norm(r)) for v,r in zip([exact,f],refs)]
        assert max(errors)<1e-12
    result=dict(feature_forcing_autodiff_relative_error=ad_error,torch_numpy_scalar_relative_errors=errors,
                samples=len(pts),includes_gaussian_centers=True,includes_tanh_center_and_interface=True)
    (args.modes.parent/'torch_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
