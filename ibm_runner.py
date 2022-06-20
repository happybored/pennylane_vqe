import pennylane as qml

import qiskit
import qiskit.providers.aer.noise as noise
from qiskit import Aer
import pennylane2qiskit as pq
dev = qml.device('qiskit.aer', wires=2)
# dev = qml.device('qiskit.aer', wires=2, backend='aer_simulator_statevector')
# dev = qml.device(
#     'qiskit.aer',
#     wires=2,
#     backend='unitary_simulator',
#     validation_threshold=1e-6
# )
dev = qml.device('qiskit.ibmq.circuit_runner', wires=2, backend='ibmq_qasm_simulator', shots=8192)
dev.capabilities()['backend']

@qml.qnode(dev)
def circuit(x, y, z):
    qml.RZ(z, wires=[0])
    qml.RY(y, wires=[0])
    qml.RX(x, wires=[0])
    qml.CNOT(wires=[0, 1])
    return qml.expval()

print(circuit(0.2, 0.1, 0.3))





# Error probabilities
prob_1 = 0.001  # 1-qubit gate
prob_2 = 0.01   # 2-qubit gate

# Depolarizing quantum errors
error_1 = noise.depolarizing_error(prob_1, 1)
error_2 = noise.depolarizing_error(prob_2, 2)

# Add errors to noise model
noise_model = noise.NoiseModel()
noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

# Create a PennyLane device
dev = qml.device('qiskit.aer', wires=2, noise_model=noise_model)

# Create a PennyLane quantum node run on the device
@qml.qnode(dev)
def circuit(x, y, z):
    qml.RZ(z, wires=[0])
    qml.RY(y, wires=[0])
    qml.RX(x, wires=[0])
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(wires=1))

# Result of noisy simulator
print(circuit(0.2, 0.1, 0.3))
