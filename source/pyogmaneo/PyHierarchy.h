// ----------------------------------------------------------------------------
//  PyOgmaNeo
//  Copyright(c) 2016-2019 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeo is licensed to you under the terms described
//  in the PYOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyConstructs.h"
#include "PyComputeProgram.h"
#include "PyIntBuffer.h"
#include <ogmaneo/Hierarchy.h>
#include <fstream>

namespace pyogmaneo {
const int _inputTypeNone = 0;
const int _inputTypeAct = 1;

struct PyLayerDesc {
    PyInt3 _hiddenSize;

    int _scRadius;
    int _aRadius;

    int _ticksPerUpdate;
    int _temporalHorizon;

    int _historyCapacity;

    PyLayerDesc()
    :
    _hiddenSize(4, 4, 16),
    _scRadius(2),
    _aRadius(2),
    _ticksPerUpdate(2),
    _temporalHorizon(2),
    _historyCapacity(8)
    {}

    PyLayerDesc(
        const PyInt3 &hiddenSize,
        int scRadius,
        int aRadius,
        int ticksPerUpdate,
        int temporalHorizon,
        int historyCapacity)
    :
    _hiddenSize(hiddenSize),
    _scRadius(scRadius),
    _aRadius(aRadius),
    _ticksPerUpdate(ticksPerUpdate),
    _temporalHorizon(temporalHorizon),
    _historyCapacity(historyCapacity)
    {}
};

class PyHierarchy {
private:
    std::vector<PyInt3> _inputSizes;

    ogmaneo::Hierarchy _h;

public:
    PyHierarchy(
        PyComputeSystem &cs,
        PyComputeProgram &prog,
        const std::vector<PyInt3> &inputSizes,
        const std::vector<int> &inputTypes,
        const std::vector<PyLayerDesc> &layerDescs
    );

    PyHierarchy(
        PyComputeSystem &cs,
        PyComputeProgram &prog,
        const std::string &name
    );

    void step(
        PyComputeSystem &cs,
        const std::vector<PyIntBuffer> &inputCs,
        float reward,
        bool learn = true);

    void save(
        PyComputeSystem &cs,
        const std::string &name
    ) {
        std::ofstream os(name, std::ios::binary);
        _h.writeToStream(cs._cs, os);
    }

    int getNumLayers() const {
        return _h.getNumLayers();
    }

    PyIntBuffer getActionCs(
        int i
    ) const {
        PyIntBuffer buf;
        buf._size = _inputSizes[i].x * _inputSizes[i].y;
        buf._buf = _h.getActionCs(i);

        return buf;
    }

    bool getUpdate(
        int l
    ) const {
        return _h.getUpdate(l);
    }

    int getTicks(
        int l
    ) const {
        return _h.getTicks(l);
    }

    int getTicksPerUpdate(
        int l
    ) const {
        return _h.getTicksPerUpdate(l);
    }

    int getNumVisibleLayers(
        int l
    ) {
        return _h.getALayers(l).size();
    }

    bool visibleLayerExists(
        int l,
        int v
    ) {
        return _h.getALayers(l)[v] != nullptr;
    }

    void setSCAlpha(
        int l,
        float alpha
    ) {
        _h.getSCLayer(l)._alpha = alpha;
    }

    float getSCAlpha(
        int l
    ) const {
        return _h.getSCLayer(l)._alpha;
    }

    void setSCExplainIters(
        int l,
        int explainIters
    ) {
        _h.getSCLayer(l)._explainIters = explainIters;
    }

    int getSCExplainIters(
        int l
    ) const {
        return _h.getSCLayer(l)._explainIters;
    }

    void setAAlpha(
        int l,
        int v,
        float alpha
    ) {
        assert(_h.getALayers(l)[v] != nullptr);
        
        _h.getALayers(l)[v]->_alpha = alpha;
    }

    float getAAlpha(
        int l,
        int v
    ) const {
        assert(_h.getALayers(l)[v] != nullptr);
        
        return _h.getALayers(l)[v]->_alpha;
    }

    void setABeta(
        int l,
        int v,
        float beta
    ) {
        assert(_h.getALayers(l)[v] != nullptr);
        
        _h.getALayers(l)[v]->_beta = beta;
    }

    float getABeta(
        int l,
        int v
    ) const {
        assert(_h.getALayers(l)[v] != nullptr);
        
        return _h.getALayers(l)[v]->_beta;
    }

    void setAGamma(
        int l,
        int v,
        float gamma
    ) {
        assert(_h.getALayers(l)[v] != nullptr);
        
        _h.getALayers(l)[v]->_gamma = gamma;
    }

    float getAGamma(
        int l,
        int v
    ) const {
        assert(_h.getALayers(l)[v] != nullptr);
        
        return _h.getALayers(l)[v]->_gamma;
    }

    void setAEpsilon(
        int l,
        int v,
        float epsilon
    ) {
        assert(_h.getALayers(l)[v] != nullptr);
        
        _h.getALayers(l)[v]->_epsilon = epsilon;
    }

    float getAEpsilon(
        int l,
        int v
    ) const {
        assert(_h.getALayers(l)[v] != nullptr);
        
        return _h.getALayers(l)[v]->_epsilon;
    }
};
} // namespace pyogmaneo