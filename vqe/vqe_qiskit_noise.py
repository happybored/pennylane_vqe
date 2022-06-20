import numpy as np
import pylab
import copy
from qiskit import BasicAer,IBMQ
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.drivers import PySCFDriver
from qiskit.chemistry.core import Hamiltonian, QubitMappingType
import sys

from qiskit import Aer
import os
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeJakarta
device_backend = FakeJakarta()

vqe_energies = []
hf_energies = []
exact_energies = []
real_vqe_energies = []
noisy_vqe_energies = []

molecule = 'H .0 .0 -{0}; Li .0 .0 {0}'
distances = np.arange(0.5,4.5,1.0)


def store_intermediate_result(eval_count, parameters, mean, std):
    print(eval_count)
    # print(parameters,eval_count,mean)
    # counts.append(eval_count)
    # values.append(mean)

backend = Aer.get_backend('aer_simulator')
counts1 = []
values1 = []
noise_model = None
device = QasmSimulator.from_backend(device_backend)
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)
basis_gates = noise_model.basis_gates

print(noise_model)
print()



for i,d in enumerate(distances):
  print("step",i)

  #setup exp
  driver = PySCFDriver(molecule.format(d/2), basis='sto3g',)
  qmolecule = driver.run()
  operator = Hamiltonian(qubit_mapping=QubitMappingType.PARITY,
                         two_qubit_reduction=True, freeze_core=True,
                         orbital_reduction=[-3,-2])
  
  qubit_op, aux_ops = operator.run(qmolecule)

  #exact res
  exact_result = NumPyMinimumEigensolver(qubit_op,aux_operators=aux_ops).run()
  exact_result = operator.process_algorithm_result(exact_result)

  #VQE
  optimizer = SLSQP(maxiter=1000)
  initial_state = HartreeFock(operator.molecule_info['num_orbitals'],
                               operator.molecule_info['num_particles'],
                               qubit_mapping=operator._qubit_mapping,
                               two_qubit_reduction=operator._two_qubit_reduction)
  var_form = UCCSD(num_orbitals=operator.molecule_info['num_orbitals'],
                   num_particles=operator.molecule_info['num_particles'],
                   initial_state=initial_state,
                   qubit_mapping=operator._qubit_mapping,
                   two_qubit_reduction=operator._two_qubit_reduction)
  
  # print(var_form)

  # sys.exit(0)

  algo = VQE(qubit_op, var_form, optimizer, aux_operators=aux_ops)

  vqe_result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
  vqe_result = operator.process_algorithm_result(vqe_result)

  # print(vqe_result)
  # sys.exit(0)



  # #VQE Real Machine
  # real_vqe_result = algo.run(QuantumInstance(backend))
  # real_vqe_result = operator.process_algorithm_result(real_vqe_result)


  noisy_algo = VQE(qubit_op, var_form, optimizer, aux_operators=aux_ops)
  # noisy_algo = VQE(qubit_op, var_form, optimizer, aux_operators=aux_ops, callback=store_intermediate_result)
  
  noisy_vqe_result = noisy_algo.run(QuantumInstance(backend, coupling_map=coupling_map, noise_model=noise_model))
  noisy_vqe_result = operator.process_algorithm_result(noisy_vqe_result)



  exact_energies.append(exact_result.energy)
  vqe_energies.append(vqe_result.energy)
  hf_energies.append(vqe_result.hartree_fock_energy)
  noisy_vqe_energies.append(noisy_vqe_result.energy)



