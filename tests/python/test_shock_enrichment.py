# SPDX-License-Identifier: GPL-3.0-or-later
"""Coupled mass inverse, moving-space conservation, and front detection."""
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def enrichment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'examples'/'pde'))
    return importlib.import_module('shock_tube_enrichment')


@pytest.mark.parametrize("scales", [(1.,), (.25, .5, 1.)])
def test_enriched_mass_and_wall_conditions(enrichment, scales):
    base = enrichment.BSPFNavierStokes(33, mu=.001, quadrature=12)
    space = enrichment.EnrichedSpace(base, np.repeat([.43, .67], len(scales)),
        (np.array([.012, .018])[:, None]*scales).ravel())
    identity = np.eye(space.n+space.k)
    V = space.evaluate(identity)
    mass = V.T@(base.weights[:, None]*V)
    rng = np.random.default_rng(12)
    for interior in (False, True):
        free = np.r_[np.arange(1, space.n-1), np.arange(space.n, space.n+space.k)] if interior else np.arange(len(identity))
        rhs = rng.normal(size=(len(free), 3))
        np.testing.assert_allclose(space.mass(rhs, interior), np.linalg.solve(mass[np.ix_(free, free)], rhs), rtol=1e-8, atol=1e-8)
    wall = space.at(np.array([0., 1.]), identity)
    np.testing.assert_allclose(wall[:, space.n:], 0, atol=1e-9)


def test_conservative_remap_and_rhs(enrichment):
    base = enrichment.BSPFNavierStokes(33, mu=.001, quadrature=12)
    old = enrichment.EnrichedSpace(base, [.5], [.012])
    new = enrichment.EnrichedSpace(base, [.51, .65], [.015, .02])
    state = old.fit(enrichment.initial_state(base.q, .02))
    state2 = new.fit(old.evaluate(state))
    np.testing.assert_allclose((old.integral@state)[[0, 2]], (new.integral@state2)[[0, 2]], atol=1e-11)
    np.testing.assert_allclose((new.integral@new.rhs(state2))[[0, 2]], 0, atol=1e-10)


def test_sensor_follows_unknown_front(enrichment):
    x = np.linspace(0, 1, 2001)
    for location in (.3, .7):
        left = .5*(1-np.tanh((x-location)/.01))
        samples = np.column_stack((.125+.875*left, np.zeros_like(x), (.1+.9*left)/.4))
        centers, widths = enrichment.detect(x, samples, 1/64, count=1)
        assert abs(centers[0]-location) < .001
        assert .005 < widths[0] < .02


def test_short_adaptive_evolution(enrichment):
    result = enrichment.run_enriched(33, .001, .006, .02)
    assert len(result['updates']) >= 3
    assert result['summary']['max_relative_mass_drift'] < 1e-10
    assert result['summary']['max_relative_energy_drift'] < 1e-10
    assert result['summary']['rejected_stages'] == 0


def test_local_spline_space_and_retained_refinement(enrichment):
    spline = importlib.import_module('shock_tube_spline_enrichment')
    base = enrichment.BSPFNavierStokes(65, mu=.0001, quadrature=32)
    factory = spline.LocalRefinement(refinement=4)
    old = factory(base, [.5], [.01])
    old_indices = factory.indices.copy()
    new = factory(base, [.52], [.01])
    assert old_indices <= factory.indices
    assert len(factory.indices) > len(old_indices)
    identity = np.eye(new.n+new.k)
    V = new.evaluate(identity)
    mass = V.T@(base.weights[:, None]*V)
    rhs = np.random.default_rng(4).normal(size=(len(identity), 2))
    np.testing.assert_allclose(mass@new.mass(rhs), rhs, atol=1e-8)
    np.testing.assert_allclose(new.at(np.array([0., 1.]), identity)[:, new.n:], 0, atol=1e-8)
    state = old.fit(enrichment.initial_state(base.q, .01))
    remapped = new.fit(old.evaluate(state))
    np.testing.assert_allclose(old.integral@state, new.integral@remapped, atol=1e-10)
    np.testing.assert_allclose(new.evaluate(remapped), old.evaluate(state), atol=1e-7)
    np.testing.assert_allclose((new.integral@new.rhs(remapped))[[0, 2]], 0, atol=1e-9)


def test_nested_enrichment_preserves_arbitrary_old_modes(enrichment):
    spline = importlib.import_module('shock_tube_spline_enrichment')
    base = enrichment.BSPFNavierStokes(65, mu=.0001, quadrature=32)
    factory = spline.LocalRefinement(refinement=8)
    old = factory(base, [.5], [.01])
    rng = np.random.default_rng(5)
    state = rng.normal(size=(old.n+old.k, 3))
    state[[0, old.n-1], 1] = 0
    for center in (.52, .55, .58):
        new = factory(base, [center], [.01])
        samples = old.evaluate(state)
        remapped = new.fit(samples)
        np.testing.assert_allclose(new.evaluate(remapped), samples, atol=2e-8)
        old, state = new, remapped


def test_sparse_enrichment_matches_dense_operators(enrichment):
    spline=importlib.import_module('shock_tube_spline_enrichment')
    base=enrichment.BSPFNavierStokes(65,mu=.0001,quadrature=32)
    dense=spline.LocalRefinement(4,space_type=spline.SplineSpace)
    sparse=spline.LocalRefinement(4,space_type=spline.SparseSplineSpace)
    for center in (.5,.52,.55):
        d=dense(base,[center],[.01]); s=sparse(base,[center],[.01])
        assert d.k==s.k
        c=d.fit(enrichment.initial_state(base.q,.02))
        for derivative in (0,1):
            np.testing.assert_allclose(s.evaluate(c,derivative),d.evaluate(c,derivative),rtol=1e-7,atol=1e-7)
        values=np.column_stack((np.sin(2*np.pi*base.q),np.cos(3*base.q)))
        np.testing.assert_allclose(s.project(values),d.project(values),atol=1e-7)
        np.testing.assert_allclose(s.rhs(c),d.rhs(c),rtol=1e-5,atol=2e-6)
        assert abs(s.radius/d.radius-1)<1e-6
        assert not hasattr(s,'H') and not hasattr(s,'Hx')


def test_sparse_transfer_injects_retained_coefficients(enrichment):
    spline=importlib.import_module('shock_tube_spline_enrichment')
    base=enrichment.BSPFNavierStokes(65,mu=.0001,quadrature=32)
    factory=spline.LocalRefinement(4,space_type=spline.SparseSplineSpace)
    old=factory(base,[.5],[.01])
    assert factory(base,[.5],[.01]) is old
    state=old.fit(enrichment.initial_state(base.q,.02))
    new=factory(base,[.53],[.01])
    transferred=new.transfer_from(old,state)
    np.testing.assert_array_equal(transferred[:len(state)],state)
    np.testing.assert_array_equal(transferred[len(state):],0)
    np.testing.assert_allclose(new.evaluate(transferred),old.evaluate(state),atol=1e-9)
    np.testing.assert_allclose(new.integral@transferred,old.integral@state,atol=1e-11)


def test_no_unused_final_time_enrichment(enrichment):
    spline=importlib.import_module('shock_tube_spline_enrichment')
    result=enrichment.run_enriched(33,.001,.006,.02,interval=.002,
        space_factory=spline.LocalRefinement(2,space_type=spline.SparseSplineSpace))
    assert all(row['time']<.006-1e-13 for row in result['updates'])
    assert result['summary']['final_time']==pytest.approx(.006)
    assert result['summary']['max_relative_mass_drift']<1e-10


def test_sparse_samples_and_probe_cache(enrichment):
    spline = importlib.import_module('shock_tube_spline_enrichment')
    base = enrichment.BSPFNavierStokes(33, mu=.001, quadrature=16)
    centers = np.array([.4, .5, .6])
    spacing = .02
    dense = spline.SplineSpace(base, centers, spacing)
    sparse = spline.SparseSplineSpace(base, centers, spacing)
    x = np.array([1., .5, .34, .4, .66, 0., .5])
    values, gradients = sparse.sparse_samples(x, centers, spacing)
    expected, expected_x = dense.raw(x)
    np.testing.assert_allclose(values.toarray(), expected, atol=1e-14)
    np.testing.assert_allclose(gradients.toarray(), expected_x, atol=1e-12)
    state = np.random.default_rng(14).normal(size=(sparse.n+sparse.k, 3))
    for probes in (x, x[::-1], x):
        np.testing.assert_allclose(sparse.at(probes, state), dense.at(probes, state), atol=1e-10)
        old_cache = sparse._probe_cache
        sparse.at(probes.copy(), state)
        assert sparse._probe_cache is old_cache
    x[0] = .51
    np.testing.assert_allclose(sparse.at(x, state), dense.at(x, state), atol=1e-10)


@pytest.mark.parametrize('backend,n', [('fft',33), ('fft',129), ('dense',33)])
def test_radius_block_products_match_quadrature(enrichment, backend, n):
    spline = importlib.import_module('shock_tube_spline_enrichment')
    base = enrichment.BSPFNavierStokes(n, mu=.001, quadrature=16, backend=backend)
    class CheckedSpace(spline.SparseSplineSpace):
        def _radius_operators(self):
            actions = super()._radius_operators()
            state = np.random.default_rng(22).normal(size=(self.n+self.k, 3))
            for derivative, action in enumerate(actions):
                expected = self.project(self.evaluate(state, derivative), derivative)
                np.testing.assert_allclose(action(state), expected, rtol=2e-8, atol=1e-7)
            return actions
    spline.LocalRefinement(2, space_type=CheckedSpace)(base, [.5], [.02])
    assert hasattr(base, '_radius_base_blocks') == (backend == 'fft' and n <= 128)


def test_fused_mass_rates_and_retained_gram(enrichment):
    spline = importlib.import_module('shock_tube_spline_enrichment')
    base = enrichment.BSPFNavierStokes(33, mu=.001, quadrature=24)
    factory = spline.LocalRefinement(4, space_type=spline.SparseSplineSpace)
    rng = np.random.default_rng(37)
    for center in (.5, .55, .6):
        space = factory(base, [center], [.02])
        size = space.n+space.k
        phi = space.evaluate(np.eye(size))
        mass = phi.T@(base.weights[:, None]*phi)
        np.testing.assert_allclose(space.small, mass[space.n:, space.n:], atol=1e-9)
        load = rng.normal(size=(size, 3))
        rate = space.solve_rates(load)
        free = np.r_[np.arange(1, space.n-1), np.arange(space.n, size)]
        np.testing.assert_array_equal(rate[[0, space.n-1], 1], 0)
        np.testing.assert_allclose(mass@rate[:, (0, 2)], load[:, (0, 2)], atol=1e-8)
        np.testing.assert_allclose((mass@rate)[free, 1], load[free, 1], atol=1e-8)
        np.testing.assert_allclose(rate, enrichment.EnrichedSpace.solve_rates(space, load), atol=1e-8)
