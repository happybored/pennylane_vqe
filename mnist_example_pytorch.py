from code import interact
from mnist import MNIST
from torch.utils.data import Dataset,DataLoader
import argparse
import torch
import pennylane as qml
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

#parameters
parser = argparse.ArgumentParser(description='PyTorch-PennyLa-Qiskit-Mnist-Example')

parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=12, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()


#classical device
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.manual_seed(args.seed)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#dataset
dataset = MNIST(
    root='./mnist_data',
    train_valid_split_ratio=[0.9, 0.1],
    digits_of_interest= [3,6],
    resize=4,
    fashion= False,
    n_train_samples= 2000,
    n_valid_samples= 200,
    n_test_samples = 10)

train_db = dataset['train']
train_data = DataLoader(train_db, batch_size=args.batch_size, shuffle=True)
valid_db = dataset['valid']
valid_data = DataLoader(valid_db, batch_size=args.test_batch_size, shuffle=True)
test_db = dataset['test']
test_data = DataLoader(valid_db, batch_size=args.test_batch_size, shuffle=True)



n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# @qml.batch_input(argnum=0)
@qml.qnode(dev,interface='torch')
def qnode(inputs, weights):
    # qml.templates.AngleEmbedding(inputs[0:4], wires=range(n_qubits),rotation='Y')
    # qml.templates.AngleEmbedding(inputs[4:8], wires=range(n_qubits),rotation='X')
    # qml.templates.AngleEmbedding(inputs[8:12], wires=range(n_qubits),rotation='Z')
    # qml.templates.AngleEmbedding(inputs[12:16], wires=range(n_qubits),rotation='Y')
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits),normalize= True)
    # qml.BasicEntanglerLayers(weights=weights, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))
    # return qml.probs(wires=[0, 1,2,3])


# weight_shapes = {"weights": (4, 4)}
weight_shapes = {"weights": (3,4,3)}
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes).to(device)

class QFCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qlayer
    def forward(self, x):
        bsz = x.shape[0]
        x = x.view(bsz,16)
        x = self.q_layer(x)
        # print(x.shape)
        x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = F.log_softmax(x, dim=1)
        return x


model = QFCModel()
optimizer = optim.Adam(model.parameters(), lr=0.1,weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

def train(train_data, model, device, optimizer):
    target_all = []
    output_all = []
    
    for batch_idx, (data, target) in enumerate(train_data):
        inputs = data.to(device)
        targets = target.to(device)

        outputs = model(inputs)

        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        target_all.append(targets)
        output_all.append(outputs)

    target_all = torch.cat(target_all, dim=0)
    output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size

    return accuracy

def valid_test(valid_data, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_data):
            inputs = data.to(device)
            targets = target.to(device)
            outputs = model(inputs)

            target_all.append(targets)
            output_all.append(outputs)

        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    return accuracy




best_acc=0
for epoch in range(1, args.epochs + 1):
    train(train_data, model, device, optimizer)
    acc = valid_test(test_data, model, device)
    if best_acc<acc:
        best_acc = acc
    print(f'Epoch {epoch}: current acc = {acc},best acc = {best_acc}')

    scheduler.step()