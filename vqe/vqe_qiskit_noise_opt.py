import numpy as np
import pylab
import copy
from qiskit import BasicAer,IBMQ
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from myvqe import VQE


from qiskit.aqua.components.optimizers import SLSQP
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD,UVCC
from qiskit.chemistry.drivers import PySCFDriver
from qiskit.chemistry.core import Hamiltonian, QubitMappingType
import sys
from qiskit.ignis.mitigation import MeasurementFilter,CompleteMeasFitter
from qiskit import Aer
import os
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeJakarta
from qiskit.algorithms.optimizers import SPSA
import math
# vqe_energies = []
# hf_energies = []
# real_vqe_energies = []

exact_energies = []
noisy_vqe_energies = []

compression_exact_energies = []
compression_noisy_vqe_energies = []


molecule = 'H .0 .0 -{0}; Li .0 .0 {0}'.format(3.0)
# molecule = 'H {} {} {}; O {} {} {};H {} {} {}'.format(-0.0399, -0.0038, 0.0, 1.5780, 0.8540, 0.0, 2.7909, -0.5159, 0.0)


def get_fixing_parameters(weight,regu_val = 6.2831853071796/4 ):
    t1 = np.floor(weight/regu_val) 
    t2 = weight % (regu_val)
    t1[t2>regu_val/2] = t1[t2>regu_val/2] + 1
    t2[t2>regu_val/2] = regu_val/2 - t2[t2>regu_val/2]
    fixing_paras = t1*regu_val
    fixing_paras = np.zeros_like(weight)
    return fixing_paras


def get_fixing_abs(weight,fix_para):
    fixing_abs= np.abs(weight-fix_para)
    return fixing_abs

def compression(weight,percent=50):
    fixing_para = get_fixing_parameters(weight)
    weight_temp = get_fixing_abs(weight,fixing_para)
    percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
    under_threshold = weight_temp < percentile
    weight[under_threshold] = fixing_para[under_threshold]
    return weight

def store_intermediate_result(vqe,eval_count, parameters, mean, std):
    # print('intermediate res:\n')
    # print('step {}, original Energy:{}'.format(eval_count,mean))
    # print(parameters)
    # compress
    parameters = compression(parameters)
    means = vqe.manual_energy_evaluation(parameters)
    print('step {}, after compression, Energy:{}'.format(eval_count,means))
    exact_energies.append(mean)
    compression_exact_energies.append(means)

def store_intermediate_result_noise(vqe,eval_count, parameters, mean, std):

    # print('step {}, original Energy:{}'.format(eval_count,mean))
    # print(parameters)
    # compress
    parameters = compression(parameters)
    means = vqe.manual_energy_evaluation(parameters)
    print('step {}, after compression, Energy:{}'.format(eval_count,means))
    noisy_vqe_energies.append(mean)
    compression_noisy_vqe_energies.append(means)


backend = Aer.get_backend('aer_simulator')
counts1 = []
values1 = []
noise_model = None
device_backend = FakeJakarta()
seed =17


device = QasmSimulator.from_backend(device_backend)
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)
basis_gates = noise_model.basis_gates
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed,
                     coupling_map=coupling_map, noise_model=noise_model,
                     measurement_error_mitigation_cls=CompleteMeasFitter,
                     cals_matrix_refresh_period=30)

qi2 =QuantumInstance(backend, coupling_map=coupling_map, noise_model=noise_model)
print('noise_model:\n',noise_model)
print()



#setup

# molecule 2 molecule Hamiltonian
driver = PySCFDriver(molecule, basis='sto3g')
qmolecule = driver.run()

# Molecule Hamiltonian 2 Qubit Hamiltonian
operator = Hamiltonian(qubit_mapping=QubitMappingType.PARITY ,two_qubit_reduction=True, freeze_core=True,orbital_reduction=[-3,-2])
# operator = Hamiltonian(qubit_mapping=QubitMappingType.JORDAN_WIGNER ,two_qubit_reduction=True, freeze_core=True,orbital_reduction=[-3,-2])
qubit_op, aux_ops = operator.run(qmolecule)


#exact res HX = lambda*X
exact_result = NumPyMinimumEigensolver(qubit_op,aux_operators=aux_ops).run()
print('exact_result:\n',exact_result)
exact_result_mol = operator.process_algorithm_result(exact_result)
print('exact_result:\n',exact_result_mol)


#VQE
optimizer = SPSA()
initial_state = HartreeFock(operator.molecule_info['num_orbitals'],
                             operator.molecule_info['num_particles'],
                             qubit_mapping=operator._qubit_mapping,
                             two_qubit_reduction=operator._two_qubit_reduction)


var_form = UCCSD(num_orbitals=operator.molecule_info['num_orbitals'],
                 num_particles=operator.molecule_info['num_particles'],
                 initial_state=initial_state,
                 qubit_mapping=operator._qubit_mapping,
                 two_qubit_reduction=operator._two_qubit_reduction)

# print('initial_state:\n',initial_state)

# print('var_form:\n',var_form)


algo = VQE(qubit_op, var_form, optimizer, aux_operators=aux_ops,callback=store_intermediate_result,max_evals_grouped =30)


vqe_result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
print('vqe_result:\n',vqe_result)

vqe_result_molecule = operator.process_algorithm_result(vqe_result)
print('vqe_result_molecule:\n',vqe_result_molecule)

print(vqe_result)
print('algo optimal_circuit:\n')

print(algo.get_optimal_circuit().decompose().draw())




# #VQE Real Machine
# real_vqe_result = algo.run(QuantumInstance(backend))
# real_vqe_result = operator.process_algorithm_result(real_vqe_result)
# noisy_algo = VQE(qubit_op, var_form, optimizer, aux_operators=aux_ops)
noisy_algo = VQE(qubit_op, var_form, optimizer, aux_operators=aux_ops, callback=store_intermediate_result_noise)

noisy_vqe_result = noisy_algo.run(qi)
print('noisy_vqe_result:\n',noisy_vqe_result)

noisy_vqe_result_molecule = operator.process_algorithm_result(noisy_vqe_result)
print('noisy_vqe_result_molecule:\n',noisy_vqe_result_molecule)




# pylab.plot(distances, hf_energies, label='Hartree-Fock')
# pylab.plot(distances, vqe_energies, 'o', label='vqe')
pylab.plot(range(0,len(exact_energies)), exact_energies, 'x', label='Exact')
pylab.plot(range(0,len(noisy_vqe_energies)), noisy_vqe_energies, '.', label='Noisy')
print(np.real(exact_result['eigenvalue']))
constant = np.full(max(len(exact_energies),len(noisy_vqe_energies)), np.real(exact_result['eigenvalue']))
pylab.plot(range(0,len(constant)),constant ,'.', label='classic')



pylab.xlabel('Iter')
pylab.ylabel('Sum')
pylab.title('LiH Ground State Energy')
pylab.legend(loc='upper right')
pylab.show()
