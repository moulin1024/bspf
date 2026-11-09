import numpy as np
import pytest
from .atlas import Atlas,AtlasObjective,Settings
from .deformation import ReducedDeformation
from .spectral import transform2
from .topology import quadrangulate
from .test_atlas import geometry,guide


@pytest.mark.parametrize('degree',[2,3,5,6])
def test_reduced_gradient_and_boundary_compatibility(degree):
    atlas=Atlas(quadrangulate(geometry()),6)
    obj=ReducedDeformation(AtlasObjective(atlas,guide(),Settings(degree=6)),degree)
    x=obj.pack(atlas.positions);rng=np.random.default_rng(3)
    x+=1e-4*rng.normal(size=len(x))
    value,g=obj(x);assert np.isfinite(value)
    direction=rng.normal(size=len(x));direction/=np.linalg.norm(direction);h=1e-7
    fd=(obj(x+h*direction)[0]-obj(x-h*direction)[0])/(2*h)
    assert abs(fd-g@direction)<2e-6*max(1,abs(fd))
    new=obj.unpack(x)
    assert np.max(abs(new[~atlas.free]-atlas.positions[~atlas.free]))<1e-14
    c=transform2((new-atlas.positions)[atlas.ids])
    assert np.max(abs(c[:,degree+1:,:]),initial=0)<1e-14
    assert np.max(abs(c[:,:,degree+1:]),initial=0)<1e-14
    if degree==atlas.degree:
        full_value,full_gradient=obj.full(obj.full.pack(obj.base)+x)
        assert abs(value-full_value)<1e-12
        assert np.max(abs(g-full_gradient))<1e-9
