"""Run a user-supplied geometry/Stokes factory without an embedded test case."""
import argparse
import importlib
import json
from pathlib import Path
from . import Settings, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module', help='Importable module exporting build_problem() -> (Geometry, StokesGuide)')
    parser.add_argument('--output', type=Path, required=True, help='Output basename; writes .npz and .json')
    parser.add_argument('--degree', type=int, default=8, help='Polynomial geometry-map degree')
    parser.add_argument('--solution-degree', type=int, default=24, help='Target PDE degree for operator-tail checks')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--spectral-tolerance', type=float, default=1e-5)
    parser.add_argument('--max-condition', type=float, default=20.)
    parser.add_argument('--max-scale-variation', type=float, default=30.)
    parser.add_argument('--quality-slack', type=float, default=1.05)
    parser.add_argument('--max-refinements', type=int, default=1)
    parser.add_argument('--max-blocks', type=int, default=256)
    parser.add_argument('--interface-weight', type=float, default=1.)
    parser.add_argument('--wall-orthogonality-weight', type=float, default=.1)
    parser.add_argument('--interface-spacing-weight', type=float, default=0.)
    parser.add_argument('--operator-barrier-weight', type=float, default=0.)
    parser.add_argument('--deformation-degree', type=int, default=None,
                        help='Restrict displacement degree while retaining the initial geometry map')
    parser.add_argument('--no-affine-preference', action='store_true',
                        help='Use the old block-count-first policy instead of preferring aligned affine charts')
    args = parser.parse_args()
    geometry, guide = importlib.import_module(args.module).build_problem()
    settings = Settings(degree=args.degree, solution_degree=args.solution_degree,
                        iterations=args.iterations, spectral_tolerance=args.spectral_tolerance,
                        max_condition=args.max_condition, max_scale_variation=args.max_scale_variation,
                        quality_slack=args.quality_slack, max_refinements=args.max_refinements,
                        max_blocks=args.max_blocks,prefer_affine_rectilinear=not args.no_affine_preference,
                        interface_weight=args.interface_weight,
                        wall_orthogonality_weight=args.wall_orthogonality_weight,
                        interface_spacing_weight=args.interface_spacing_weight,
                        operator_barrier_weight=args.operator_barrier_weight,
                        deformation_degree=args.deformation_degree)
    result = generate(geometry, guide, settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.atlas.save(args.output.with_suffix('.npz'), result.report)
    args.output.with_suffix('.json').write_text(json.dumps(result.report, indent=2, allow_nan=False)+'\n')
    failed = [b['block'] for b in result.report['quality']['blocks'] if b['failures']]
    print(json.dumps(dict(accepted=result.report['quality']['accepted'], blocks=len(result.atlas.ids),
                          failed_blocks=failed, optimizer=result.report['optimization']['status'])))
    return 0 if result.report['quality']['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
