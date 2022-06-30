from qiskit.providers.aer.utils import approximate_quantum_error
from qiskit.providers.aer.utils import approximate_noise_model
import numpy as np

# Import Aer QuantumError functions that will be used
from qiskit.providers.aer.noise import amplitude_damping_error
from qiskit.providers.aer.noise import reset_error
from qiskit.providers.aer.noise import pauli_error

gamma = 0.23
error = amplitude_damping_error(gamma)

results = approximate_quantum_error(error, operator_string="reset")
print("\n")
print(results)

p = (1 + gamma - np.sqrt(1 - gamma)) / 2
q = 0

print("\n")
print("Expected results:")
print("P(0) = {}".format(1-(p+q)))
print("P(1) = {}".format(p))
print("P(2) = {}".format(q))
print("\n")
gamma = 0.23
K0 = np.array([[1,0],[0,np.sqrt(1-gamma)]])
K1 = np.array([[0,np.sqrt(gamma)],[0,0]])
results = approximate_quantum_error((K0, K1), operator_string="reset")
print(results)
print("\n")
reset_to_0 = [np.array([[1,0],[0,0]]), np.array([[0,1],[0,0]])]
reset_to_1 = [np.array([[0,0],[1,0]]), np.array([[0,0],[0,1]])]
reset_kraus = (reset_to_0, reset_to_1)

print("\n")
gamma = 0.23
error = amplitude_damping_error(gamma)
results = approximate_quantum_error(error, operator_list=reset_kraus)
print(results)
