"""Generate and validate nonperiodic 3D incompressible MMS data on CUDA."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bspf-turbulence3d-mpl')
import argparse
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
from scipy.special import roots_legendre
from mms import CurvedBox, PlaneWaveMMS, random_modes


def quadrature_statistics(field, domain, order):
    cp = field.xp
    z, w = roots_legendre(order); z = cp.asarray((z+1)/2); w = cp.asarray(w/2)
    points = cp.stack(cp.meshgrid(z, z, z, indexing='ij'), axis=-1).reshape(-1, 3)
    physical, _, jac = domain.map(points, cp)
    weights = (w[:, None, None]*w[None, :, None]*w[None, None, :]).ravel()*jac
    volume = cp.sum(weights)
    v = field.evaluate(physical, derivatives=False)
    u, p = v['velocity'], v['pressure']
    mean = cp.sum(weights[:, None]*u, axis=0)/volume
    pm = cp.sum(weights*p)/volume
    covariance = (u.T@(weights[:, None]*u))/volume
    return dict(volume=float(volume), velocity_mean=cp.asnumpy(mean).tolist(),
                velocity_rms=float(cp.sqrt(cp.trace(covariance))),
                velocity_covariance=cp.asnumpy(covariance).tolist(),
                pressure_mean=float(pm), pressure_rms=float(cp.sqrt(cp.sum(weights*p*p)/volume)),
                min_jacobian=float(cp.min(jac)), max_jacobian=float(cp.max(jac)))


def normalize(field, domain, order, velocity_rms, pressure_rms):
    stats = quadrature_statistics(field, domain, order)
    mean = np.array(stats['velocity_mean']); pm = stats['pressure_mean']
    uscale = velocity_rms/np.sqrt(stats['velocity_rms']**2-mean@mean)
    pscale = pressure_rms/np.sqrt(stats['pressure_rms']**2-pm*pm)
    field.a *= uscale; field.b *= uscale; field.u_offset = field.xp.asarray(-mean*uscale)
    field.pc *= pscale; field.ps *= pscale; field.p_offset = -pm*pscale
    field._features = None
    return stats, uscale, pscale


def cpu_reference(field, points):
    """Independent complex-exponential evaluation for off-grid verification."""
    phase = np.exp(1j*((np.asarray(points)-field.center)@field.k.T))
    uc = field.a-1j*field.b; pc = field.pc-1j*field.ps
    k2 = np.sum(field.k**2, axis=1)
    u = (phase@uc).real+field.u_offset
    p = (phase@pc).real+field.p_offset
    grad = (phase@(1j*uc[:, :, None]*field.k[:, None, :]).reshape(-1, 9)).real.reshape(-1, 3, 3)
    lap = (phase@(k2[:, None]*uc)).real
    gradp = (phase@(1j*pc[:, None]*field.k)).real
    stokes = field.viscosity*lap+gradp
    return dict(velocity=u, pressure=p, velocity_gradient=grad, vector_poisson_forcing=lap,
                pressure_gradient=gradp, pressure_poisson_forcing=(phase@(k2*pc)).real,
                stokes_forcing=stokes, navier_stokes_forcing=stokes+np.einsum('nj,nij->ni', u, grad))


def verify(field, domain, modes_path, order):
    cp = field.xp; host = PlaneWaveMMS.load(modes_path)
    rng = np.random.default_rng(7142)
    reference_points = rng.random((24, 3)); points = domain.map(reference_points)[0]
    gpu = {k: cp.asnumpy(v) for k, v in field.evaluate(cp.asarray(points)).items()}
    exact = cpu_reference(host, points)
    errors = {k: float(np.max(abs(gpu[k]-v))/max(1., np.max(abs(v)))) for k, v in exact.items()}
    if max(errors.values()) > 2e-11:
        raise RuntimeError(f'GPU/CPU analytic evaluation mismatch: {errors}')
    gradient = gpu['velocity_gradient']
    divergence = np.trace(gradient, axis1=1, axis2=2)
    relative_div = float(np.linalg.norm(divergence)/np.linalg.norm(gradient))
    if relative_div > 2e-13: raise RuntimeError('Off-grid divergence check failed')
    # Independent fourth-order finite differences at off-grid physical points.
    h = 1e-4; grad_fd = np.empty_like(gradient); minus_lap_fd = np.zeros_like(gpu['velocity'])
    pressure_grad_fd = np.empty_like(gpu['pressure_gradient'])
    for axis in range(3):
        step = np.eye(3)[axis]*h
        fm2, fm1, fp1, fp2 = [cpu_reference(host, points+s*step) for s in (-2, -1, 1, 2)]
        grad_fd[:, :, axis] = (fm2['velocity']-8*fm1['velocity']+8*fp1['velocity']-fp2['velocity'])/(12*h)
        pressure_grad_fd[:, axis] = (fm2['pressure']-8*fm1['pressure']+8*fp1['pressure']-fp2['pressure'])/(12*h)
        minus_lap_fd += (fp2['velocity']-16*fp1['velocity']+30*exact['velocity']-16*fm1['velocity']+fm2['velocity'])/(12*h*h)
    force_fd = np.einsum('nj,nij->ni', exact['velocity'], grad_fd)+pressure_grad_fd+field.viscosity*minus_lap_fd
    fd_errors = dict(gradient=float(np.linalg.norm(grad_fd-gradient)/np.linalg.norm(gradient)),
                     vector_poisson=float(np.linalg.norm(minus_lap_fd-exact['vector_poisson_forcing'])/np.linalg.norm(exact['vector_poisson_forcing'])),
                     navier_stokes=float(np.linalg.norm(force_fd-exact['navier_stokes_forcing'])/np.linalg.norm(exact['navier_stokes_forcing'])))
    if max(fd_errors.values()) > 2e-7: raise RuntimeError(f'Finite-difference derivative check failed: {fd_errors}')
    # Flux compatibility over six independent curved physical boundaries.
    z, w = roots_legendre(order); z = cp.asarray((z+1)/2); w = cp.asarray(w/2)
    uv = cp.stack(cp.meshgrid(z, z, indexing='ij'), axis=-1).reshape(-1, 2)
    weights = (w[:, None]*w[None, :]).ravel()
    fluxes = {}; absolute_flux = 0.; mismatches = {}
    for axis in range(3):
        pair = []
        for side in (0, 1):
            x, area = domain.face(axis, side, uv, cp)
            u = field.evaluate(x, derivatives=False)['velocity']
            integrand = cp.sum(u*area, axis=1)
            fluxes[f'{"xyz"[axis]}{side}'] = float(cp.sum(weights*integrand))
            absolute_flux += float(cp.sum(weights*cp.abs(integrand)))
            pair.append(u)
        mismatches['xyz'[axis]] = float(cp.sqrt(cp.sum(weights[:, None]*(pair[1]-pair[0])**2)/cp.sum(weights[:, None]*(pair[1]**2+pair[0]**2)/2)))
    net = sum(fluxes.values())
    if abs(net)/max(absolute_flux, 1e-300) > 1e-10: raise RuntimeError('Boundary mass-flux compatibility failed')
    if min(mismatches.values()) < .1: raise RuntimeError('Unexpected matching opposite boundary traces')
    return dict(gpu_cpu_relative_max_errors=errors, off_grid_relative_divergence=relative_div,
                finite_difference_relative_errors=fd_errors, boundary_face_fluxes=fluxes,
                boundary_net_flux=net, boundary_relative_net_flux=abs(net)/absolute_flux,
                opposite_face_velocity_relative_mismatch=mismatches)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=97, help='Reference sampling nodes per axis, including boundaries')
    ap.add_argument('--shell-min', type=int, default=4); ap.add_argument('--shell-max', type=int, default=32)
    ap.add_argument('--modes-per-shell', type=int, default=64); ap.add_argument('--seed', type=int, default=20261116)
    ap.add_argument('--quadrature', type=int, default=96)
    ap.add_argument('--viscosity', type=float, default=.01)
    ap.add_argument('--velocity-rms', type=float, default=1.); ap.add_argument('--pressure-rms', type=float, default=.1)
    ap.add_argument('--save-gradient', action='store_true')
    args = ap.parse_args()
    if args.n < 3 or args.quadrature < 8 or not 1 <= args.shell_min < args.shell_max or args.modes_per_shell < 1:
        ap.error('Invalid grid, quadrature order, or modal band')
    if not np.isfinite([args.viscosity,args.velocity_rms,args.pressure_rms]).all() or args.viscosity < 0 or min(args.velocity_rms,args.pressure_rms) <= 0:
        ap.error('Finite nonnegative viscosity and positive RMS values required')
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()): ap.error('Use an empty output directory to preserve existing data')
    import cupy as cp
    # Require a real CUDA device; no CPU generation fallback.
    device = cp.cuda.runtime.getDeviceProperties(0)
    wall_start = time.perf_counter()
    domain = CurvedBox()
    cp.cuda.Stream.null.synchronize(); start = time.perf_counter()
    field, shell_ids, target = random_modes(cp, seed=args.seed, shell_min=args.shell_min,
                        shell_max=args.shell_max, modes_per_shell=args.modes_per_shell, viscosity=args.viscosity)
    cp.cuda.Stream.null.synchronize(); mode_seconds = time.perf_counter()-start
    print(f'GPU modes: {len(field.k)} continuous wavevectors; normalizing on curved domain', flush=True)
    raw, uscale, pscale = normalize(field, domain, args.quadrature, args.velocity_rms, args.pressure_rms)
    modes_path = args.out/'modes.npz'
    field.save(modes_path, shell_ids=cp.asnumpy(shell_ids), k_unit=np.pi, seed=args.seed, amplitudes=domain.amplitudes)
    stats = quadrature_statistics(field, domain, args.quadrature+16)
    normalization_error = max(abs(stats['velocity_rms']/args.velocity_rms-1), abs(stats['pressure_rms']/args.pressure_rms-1), np.linalg.norm(stats['velocity_mean'])/args.velocity_rms, abs(stats['pressure_mean'])/args.pressure_rms)
    if normalization_error > 1e-10: raise RuntimeError('Normalization quadrature unresolved; increase --quadrature')
    energy = cp.bincount(shell_ids, weights=.25*cp.sum(field.a**2+field.b**2, axis=1), minlength=args.shell_max+1)[args.shell_min:]
    shell_error = float(cp.max(cp.abs(energy/(target*uscale**2)-1)))
    spectrum_slope = float(np.polyfit(np.log(np.arange(args.shell_min,args.shell_max+1)), np.log(cp.asnumpy(energy)), 1)[0])
    if shell_error > 2e-12 or abs(spectrum_slope+5/3) > 2e-12: raise RuntimeError('Modal spectrum check failed')
    kdot = cp.stack((cp.sum(field.k*field.a, axis=1), cp.sum(field.k*field.b, axis=1)))
    polarization_div = float(cp.linalg.norm(kdot)/cp.sqrt(cp.sum(cp.sum(field.k**2,axis=1)*cp.sum(field.a**2+field.b**2,axis=1))))
    checks = verify(field, domain, modes_path, args.quadrature+16)
    print('Analytic, divergence, nonperiodic-trace and curved-boundary flux checks passed', flush=True)
    metadata = dict(status='generating', geometry='Q2 curved hexahedron; six physical boundaries; no periodic identification',
        map_amplitudes=domain.amplitudes.tolist(), reference_box=[[0,1]]*3, periodic=False,
        spectrum='Prescribed modal kinetic energy per radial shell, E_s proportional to s^(-5/3). Not an FFT of nonperiodic samples.',
        wavevectors='Continuous random directions and radii; not a reciprocal lattice', k_unit=float(np.pi),
        parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        gpu=device['name'].decode() if isinstance(device['name'],bytes) else device['name'],
        cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cuda_driver=cp.cuda.runtime.driverGetVersion(),
        precision='float64', rng='CuPy default_rng (XORWOW)', number_of_wavevectors=len(field.k),
        time_dependence='steady', density=1.,
        equations=dict(vector_poisson='-Delta u=f_vector_poisson', pressure_poisson='-Delta p=f_pressure_poisson',
                       stokes='-nu Delta u+grad p=f_stokes; div u=0',
                       navier_stokes='(u dot grad)u+grad p-nu Delta u=f_navier_stokes; div u=0'),
        modes_file_sha256=hashlib.sha256(modes_path.read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('mms.py'))},
        normalization_orders=[args.quadrature,args.quadrature+16], normalization_check_relative_error=float(normalization_error),
        domain_statistics=stats, normalization_scales=dict(velocity=uscale,pressure=pscale),
        modal_shells=list(range(args.shell_min,args.shell_max+1)), modal_shell_energy=cp.asnumpy(energy).tolist(),
        modal_spectrum_slope=spectrum_slope, modal_shell_max_relative_error=shell_error,
        polarization_relative_divergence=polarization_div, validation=checks,
        timings=dict(gpu_mode_generation_seconds=mode_seconds))
    def save_report(): (args.out/'report.json').write_text(json.dumps(metadata,indent=2)+'\n')
    save_report()
    n=args.n; z=cp.linspace(0,1,n); gpu_seconds=0.; io_seconds=0.; div2=0.; grad2=0.; maxdiv=0.
    with h5py.File(args.out/'fields.h5','w') as h5:
        h5.attrs['layout']='spatial axes (s,t,r), then physical Cartesian component axes; velocity_gradient[...,i,j]=du_i/dx_j'
        h5.attrs['periodic']=False; h5.attrs['viscosity']=args.viscosity
        h5.attrs['modes_file_sha256']=metadata['modes_file_sha256']
        h5.create_dataset('reference_axis',data=cp.asnumpy(z))
        datasets={}
        for first in range(0,n,4):
            ref=cp.stack(cp.meshgrid(z[first:first+4],z,z,indexing='ij'),axis=-1)
            begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record()
            x,jac,det=domain.map(ref,cp);del jac
            values=field.evaluate(x)
            gradient=values['velocity_gradient'];div=cp.trace(gradient,axis1=-2,axis2=-1)
            div2+=float(cp.sum(div*div));grad2+=float(cp.sum(gradient*gradient));maxdiv=max(maxdiv,float(cp.max(cp.abs(div))))
            if not args.save_gradient:del values['velocity_gradient']
            values.update(coordinates=x,jacobian=det)
            end.record();end.synchronize();gpu_seconds+=cp.cuda.get_elapsed_time(begin,end)/1000
            start=time.perf_counter()
            for key,value in values.items():
                host=cp.asnumpy(value)
                if key not in datasets:
                    datasets[key]=h5.create_dataset(key,shape=(n,n,n)+host.shape[3:],dtype='f8')
                datasets[key][first:first+len(host)]=host
            io_seconds+=time.perf_counter()-start
        boundary=h5.create_group('boundary')
        uv=cp.stack(cp.meshgrid(z,z,indexing='ij'),axis=-1)
        for axis in range(3):
            for side in (0,1):
                x,area=domain.face(axis,side,uv,cp);data=field.evaluate(x,derivatives=False)
                data.update(coordinates=x,outward_normal=area/cp.linalg.norm(area,axis=-1)[...,None])
                group=boundary.create_group(f'{"xyz"[axis]}{side}')
                for key,value in data.items():group.create_dataset(key,data=cp.asnumpy(value))
    if np.sqrt(div2/grad2)>2e-13:raise RuntimeError('Volume sample divergence check failed')
    metadata['validation'].update(volume_sample_relative_divergence=float(np.sqrt(div2/grad2)),volume_sample_max_abs_divergence=maxdiv)
    metadata['timings'].update(gpu_volume_sampling_seconds=gpu_seconds,volume_transfer_and_write_seconds=io_seconds,total_wall_seconds=time.perf_counter()-wall_start)
    metadata['dataset_bytes']=(args.out/'fields.h5').stat().st_size
    metadata['status']='validated';save_report()
    plot(args.out,metadata)
    print(json.dumps(dict(status=metadata['status'],gpu=metadata['gpu'],shape=[n]*3,spectrum_slope=spectrum_slope,
                         relative_divergence=metadata['validation']['volume_sample_relative_divergence'],
                         relative_net_boundary_flux=checks['boundary_relative_net_flux'],seconds=metadata['timings']['total_wall_seconds'])),flush=True)


def plot(out,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with h5py.File(out/'fields.h5') as f:
        n=len(f['reference_axis']);x=f['coordinates'][:,:,n//2,:];u=f['velocity'][:,:,n//2,:];vort=f['vorticity'][:,:,n//2,:]
        faces={key:(group['coordinates'][...],group['velocity'][...]) for key,group in f['boundary'].items()}
    fig=plt.figure(figsize=(15,4.8),layout='constrained')
    ax=fig.add_subplot(131,projection='3d');speeds=[np.linalg.norm(v,axis=-1) for _,v in faces.values()]
    norm=matplotlib.colors.Normalize(min(v.min() for v in speeds),max(v.max() for v in speeds));cmap=plt.get_cmap('viridis')
    for (xyz,_),speed in zip(faces.values(),speeds):
        ax.plot_surface(xyz[...,0],xyz[...,1],xyz[...,2],facecolors=cmap(norm(speed)),rstride=max(1,n//32),cstride=max(1,n//32),shade=False,alpha=.85)
    ax.set(xlabel='x',ylabel='y',zlabel='z',title='Six curved, nonperiodic boundaries');ax.set_box_aspect((1,1,1));fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.6,label='Speed')
    ax=fig.add_subplot(132);im=ax.pcolormesh(x[...,0],x[...,1],np.linalg.norm(vort,axis=-1),shading='gouraud',cmap='magma')
    ax.set_aspect('equal');ax.set(xlabel='x',ylabel='y',title='Vorticity magnitude on r=0.5 slice');fig.colorbar(im,ax=ax,shrink=.7)
    ax=fig.add_subplot(133);s=np.array(report['modal_shells']);energy=np.array(report['modal_shell_energy'])
    ax.loglog(s,energy,'o',ms=4,label='Transverse-wave coefficients');ax.loglog(s,energy[0]*(s/s[0])**(-5/3),'--',label=r'$s^{-5/3}$ target')
    ax.set(xlabel='Radial mode shell (|k| / π)',ylabel='Modal kinetic energy',title='Prescribed modal spectrum');ax.grid(alpha=.25,which='both');ax.legend(fontsize=8)
    fig.suptitle('3D incompressible manufactured field — no periodic domain or lattice')
    fig.savefig(out/'preview.png',dpi=180);fig.savefig(out/'preview.pdf');plt.close(fig)

if __name__=='__main__':main()
