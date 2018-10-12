// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyActor.h"

using namespace pyogmaneo;

PyActor::PyActor(PyComputeSystem &cs, PyComputeProgram &prog, const PyInt3 &hiddenSize, const std::vector<PyAVisibleLayerDesc> &visibleLayerDescs) {
    _alpha = _a._alpha;
    _gamma = _a._gamma;

    _visibleLayerDescs = visibleLayerDescs;

    std::vector<ogmaneo::Actor::VisibleLayerDesc> clVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        clVisibleLayerDescs[v]._size = cl_int3{ visibleLayerDescs[v]._size.x, visibleLayerDescs[v]._size.y, visibleLayerDescs[v]._size.z };
        clVisibleLayerDescs[v]._radius = visibleLayerDescs[v]._radius;
    }

    _a.createRandom(cs._cs, prog._prog, cl_int3{ hiddenSize.x, hiddenSize.y, hiddenSize.z }, clVisibleLayerDescs, cs._rng);
}

void PyActor::step(PyComputeSystem &cs, const std::vector<PyIntBuffer> &visibleCs, const PyIntBuffer &targetCs, float reward, bool learn) {
    _a._alpha = _alpha;
    _a._gamma = _gamma;

    std::vector<cl::Buffer> clVisibleCs(visibleCs.size());

    for (int v = 0; v < visibleCs.size(); v++)
        clVisibleCs[v] = visibleCs[v]._buf;

    _a.step(cs._cs, clVisibleCs, targetCs._buf, reward, learn);
}