from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np
import pennylane as qml

dev = qml.device('default.qubit', wires=2)

theta = Parameter('θ')

qc = QuantumCircuit(2)
qc.rz(theta, [0])
qc.rx(theta, [0])
qc.cx(0, 1)

@qml.qnode(dev)
def quantum_circuit_with_loaded_subcircuit(x):
    qml.from_qiskit(qc)({theta: x})
    return qml.expval(qml.PauliZ(0))

angle = np.pi/2
result = quantum_circuit_with_loaded_subcircuit(angle)
print(result)