import pennylane as qml
from pennylane_cirq import ops as cirq_ops 
from pennylane import numpy as np
import torch 

dev = qml.device("cirq.mixedsimulator", wires=2, shots=1000)


@qml.qnode(dev,interface='torch')
def circuit(gate_params, noise_param=0.0):
    qml.RX(gate_params[0], wires=0)
    qml.RY(gate_params[1], wires=0)
    cirq_ops.Depolarize(noise_param, wires=0)
    return qml.expval(qml.PauliZ(0))

gate_pars = torch.tensor([0.011, 0.012], requires_grad=True)

print("Expectation value:", circuit(gate_pars))

# declare the cost functions to be optimized
def cost(x):
    return circuit(x, noise_param=0.0)

def noisy_cost(x):
    return circuit(x, noise_param=0.3)

# initialize the optimizer
# opt = qml.GradientDescentOptimizer(stepsize=0.4)
opt = torch.optim.Adam([gate_pars], lr = 0.1)


# set the number of steps
steps = 10

def closure():
    opt.zero_grad()
    loss = noisy_cost(gate_pars)
    loss.backward()
    return loss

for i in range(steps):
    opt.step(closure)
    print(gate_pars)


# for i in range(steps):
#     # update the circuit parameters
#     # we can optimize both in the same training loop
#     params = opt.step(cost, params)
#     noisy_circuit_params = opt.step(noisy_cost, noisy_circuit_params)

#     if (i + 1) % 5 == 0:
#         print("Step {:5d}. Cost: {: .7f}; Noisy Cost: {: .7f}".
#               format(i + 1,
#                      cost(params),
#                      noisy_cost(noisy_circuit_params)))

# print("\nOptimized rotation angles (noise-free case):")
# print("({: .7f}, {: .7f})".format(*params))
# print("Optimized rotation angles (noisy case):")
# print("({: .7f}, {: .7f})".format(*noisy_circuit_params))
