import pennylane as qml
from pennylane_cirq import ops as cirq_ops 
from pennylane import numpy as np
dev = qml.device("cirq.mixedsimulator", wires=2, shots=1000)


@qml.qnode(dev)
def circuit(gate_params, noise_param=0.0):
    qml.RX(gate_params[0], wires=0)
    qml.RY(gate_params[1], wires=0)
    cirq_ops.Depolarize(noise_param, wires=0)
    return qml.expval(qml.PauliZ(0))

gate_pars = [0.54, 0.12]
print("Expectation value:", circuit(gate_pars))

# declare the cost functions to be optimized
def cost(x):
    return circuit(x, noise_param=0.0)

def noisy_cost(x):
    return circuit(x, noise_param=0.3)

# initialize the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# set the number of steps
steps = 100
# set the initial parameter values
init_params = np.array([0.011, 0.055], requires_grad=True)
noisy_circuit_params = init_params
params = init_params

for i in range(steps):
    # update the circuit parameters
    # we can optimize both in the same training loop
    params = opt.step(cost, params)
    noisy_circuit_params = opt.step(noisy_cost, noisy_circuit_params)

    if (i + 1) % 5 == 0:
        print("Step {:5d}. Cost: {: .7f}; Noisy Cost: {: .7f}".
              format(i + 1,
                     cost(params),
                     noisy_cost(noisy_circuit_params)))

print("\nOptimized rotation angles (noise-free case):")
print("({: .7f}, {: .7f})".format(*params))
print("Optimized rotation angles (noisy case):")
print("({: .7f}, {: .7f})".format(*noisy_circuit_params))
