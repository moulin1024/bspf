# BSPF Docs

This directory contains the package-level documentation for the in-repo `bspf`
package.

## Read First

- [README.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/README.md): install, quick start, and current project status
- [api.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/api.md): current public API summary
- [design.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/design.md): package architecture and numerical design
- [compatibility_strategy.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/compatibility_strategy.md): transition policy between the package API and the legacy module
- [refactor_backlog.md](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/docs/refactor_backlog.md): phased migration record

## Current Shape

The package code lives in [`src/bspf`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/src/bspf) and is organized around:

- backend selection and device validation
- uniform grids and knot generation
- spline basis and endpoint operators
- residual correction and KKT solve helpers
- operation-family modules in `ops/`
- user-facing operator classes in `operators/`

The legacy monolithic implementation [`bspf1d.py`](/Users/moulin/Library/CloudStorage/Dropbox/workspace/bspf/bspf1d.py) is still present for compatibility and regression comparison.
