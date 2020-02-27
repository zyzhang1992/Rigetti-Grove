import numpy as np
from pyquil.api import WavefunctionSimulator

from grove.pyqaoa.maxcut_qaoa_weighted import maxcut_qaoa_weighted
steps = 2
#square_ring = [(0,1),(1,2),(2,3),(3,0)]
square_ring = [(0,1,0.5),(1,2,0.5),(2,3,0.5),(3,0,0.5)]

inst = maxcut_qaoa_weighted(square_ring, steps=steps)
opt_betas, opt_gammas = inst.get_angles()

t = np.hstack((opt_betas, opt_gammas))
param_prog = inst.get_parameterized_program()
prog = param_prog(t)
wf = WavefunctionSimulator().wavefunction(prog)
wf = wf.amplitudes

for state_index in range(inst.nstates):
    print(inst.states[state_index], np.conj(wf[state_index]) * wf[state_index])
