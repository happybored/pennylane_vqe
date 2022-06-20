import torch
import pennylane as qml


def my_quantum_function(x, y):
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(y, wires=1)
    qml.CNOT(wires=[0, 2])
    return qml.probs(wires=[0, 1,2])

dev = qml.device("default.qubit", wires=3)
qnode = qml.QNode(my_quantum_function, dev)
result = qnode(0.56, 0.1)
print(result)