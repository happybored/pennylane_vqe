import torch
import pennylane as qml

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev, interface='torch')
def circuit4(phi, theta):
    qml.RX(phi[0], wires=0)
    qml.RZ(phi[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(theta, wires=0)
    return qml.expval(qml.PauliZ(0))

def cost(phi, theta):
    return torch.abs(circuit4(phi, theta) - 0.5)**2

phi = torch.tensor([0.011, 0.012], requires_grad=True)
theta = torch.tensor(0.05, requires_grad=True)

opt = torch.optim.Adam([phi, theta], lr = 0.1)

steps = 200

def closure():
    opt.zero_grad()
    loss = cost(phi, theta)
    loss.backward()
    return loss

for i in range(steps):
    opt.step(closure)


#amplitude embedding

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev,interface='torch')
def circuit(f=None):
    qml.AmplitudeEmbedding(features=f, wires=range(2))
    return qml.expval(qml.PauliZ(0))

circuit(f=[1/2, 1/2, 1/2, 1/2])
print(dev.state)


dev = qml.device('default.qubit', wires=3)

@qml.qnode(dev)
def circuit(feature_vector):
    qml.AngleEmbedding(features=feature_vector, wires=range(3), rotation='Z')
    qml.Hadamard(0)
    return qml.probs(wires=range(3))

X = [1,2,3]

print(qml.draw(circuit, expansion_strategy="device")(X))
