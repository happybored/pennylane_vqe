from qiskit import IBMQ, transpile,assemble
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator,QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.visualization import plot_histogram
from qiskit.test.mock import FakeVigo
from matplotlib import pyplot as plt
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter
from qiskit.ignis.mitigation import MeasurementFilter,CompleteMeasFitter
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit import Aer,QuantumRegister
from qiskit.aqua.utils.run_circuits import run_qobj
import sys
from qiskit.aqua.utils.measurement_error_mitigation import (get_measured_qubits_from_qobj,build_measurement_error_mitigation_qobj)
import qiskit


from mitiq import zne
from mitiq.interface.mitiq_qiskit.qiskit_utils import initialized_depolarizing_noise

device_backend = FakeVigo()
noise_model = NoiseModel.from_backend(device_backend)
coupling_map = device_backend.configuration().coupling_map



# Construct quantum circuit
circ = QuantumCircuit(3, 3)
circ.h(0)
circ.h(2)
circ.cx(0, 1)
circ.cx(1, 2)
circ.ry(0.2,1)
circ.rx(0.2,0)

circ.measure([0, 1, 2], [0, 1, 2])



shots=10000
circuit =  circ
# Convert from raw measurement counts to the expectation value
scale_factors = [1., 1.5, 2.,2.5,3.0]
folded_circuits = [
        zne.scaling.fold_gates_at_random(circuit, scale)
        for scale in scale_factors
]
job = qiskit.execute(
    experiments=folded_circuits,
    backend=qiskit.Aer.get_backend("qasm_simulator"),
    noise_model=noise_model,
    basis_gates=noise_model.basis_gates,
    optimization_level=0,  # Important to preserve folded gates.
    shots=shots,
)
# Check that the circuit depth is (approximately) scaled as expected
for j, c in enumerate(folded_circuits):
    print(f"Number of gates of folded circuit {j} scaled by: {len(c) / len(circuit):.3f}")
all_counts = [job.result().get_counts(i) for i in range(len(folded_circuits))]
expectation_values = [counts.get("000") / shots for counts in all_counts]
print(expectation_values)



zero_noise_value = zne.ExpFactory.extrapolate(scale_factors, expectation_values, asymptote=0.5)
print(f"Unmitigated result {expectation_values[0]}")
print(f"Mitigated result {zero_noise_value}")