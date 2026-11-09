"""Serial FC-Gram shock-tube resolution sweep with saved numerical reference."""
import argparse,json,sys
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples/pde'))
from fc_shock_tube import solve_fc_tube
from compiled_galerkin_step import warmup
from shock_tube_1d import primitives


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,nargs='+',default=[487,999,2023,4071])
    p.add_argument('--cfl',type=float,default=1.)
    p.add_argument('--filter-order',type=int,default=0)
    p.add_argument('--tag',default='')
    a=p.parse_args();warmup()
    data=ROOT/'benchmarks/traveling_shock/prior_art/fc_data/FCGram_data_d5_C25.mat'
    ref=np.load(ROOT/'benchmarks/shock_tube/enrichment_n65_mu0001_fine.npz')
    folder=ROOT/'benchmarks/shock_tube/fc_gram';folder.mkdir(parents=True,exist_ok=True)
    for n in a.n:
        start=perf_counter()
        result=solve_fc_tube(n,data,cfl=a.cfl,filter_order=a.filter_order,progress=True)
        result['summary']['end_to_end_seconds']=perf_counter()-start
        # The dense output grid contains every stored reference probe exactly.
        assert np.allclose(result['x'][::32],ref['x'],atol=1e-15,rtol=0)
        result['summary']['stored_reference_discrepancy']={name:float(np.max(abs(v-r)))
            for name,v,r in zip(('rho','u','p','T'),primitives(result['output'][::32]),primitives(ref['reference']))}
        prefix=folder/f'fc_{n}{a.tag}'
        prefix.with_suffix('.json').write_text(json.dumps(result['summary'],indent=2))
        np.savez(prefix.with_suffix('.npz'),**{k:v for k,v in result.items() if k not in ('summary','history')})
        print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
