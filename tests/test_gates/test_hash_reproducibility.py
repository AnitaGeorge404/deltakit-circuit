# (c) Copyright Riverlane 2020-2025.
"""Regression tests for https://github.com/Deltakit/deltakit/issues/354.

``Gate.__hash__`` and ``NoiseChannel.__hash__`` used to include the gate's
``self.__class__`` object in the hashed tuple. A class object's hash is
``object.__hash__``, which is based on the class's memory address. That
address is not fixed by ``PYTHONHASHSEED`` and differs between separate
Python processes, so any code that iterates a ``set``/``frozenset`` of gates
(for example when building a :class:`~deltakit_circuit.GateLayer` from a set
comprehension) produced a different iteration order in every process. That
nondeterministic order propagated through to the emitted Stim circuit: the
order of targets within a single Stim instruction, and the indices assigned
by :func:`~deltakit_circuit.default_qubit_mapping`, could both change from
run to run even though the logical circuit was unchanged.

These tests spawn separate subprocesses (with a fixed ``PYTHONHASHSEED``) to
verify that emission is now reproducible across process boundaries, which an
in-process test cannot demonstrate since ``id()``-based hashes are stable for
the lifetime of a single process.
"""

import hashlib
import os
import subprocess
import sys
import textwrap

import pytest

from deltakit_circuit import Qubit
from deltakit_circuit.gates import CX, H, I, X

# Script executed in a fresh subprocess. It builds a GateLayer from a `set`
# of gate instances on purpose: this mirrors how circuit-generation code
# (e.g. deltakit_explorer's CSS code builders) constructs layers, and is the
# exact pattern that turned a process-dependent gate hash into a
# process-dependent instruction/target order.
_REPRO_SCRIPT = textwrap.dedent(
    """
    import hashlib
    from deltakit_circuit import Circuit, GateLayer, Qubit
    from deltakit_circuit.gates import CX, H, I, X

    data_qubits = [Qubit((i, j)) for i in range(3) for j in range(3)]
    reset_layer = GateLayer({I(qubit) for qubit in data_qubits})
    hadamard_layer = GateLayer({H(qubit) for qubit in data_qubits})
    cx_layer = GateLayer(
        {CX(data_qubits[i], data_qubits[i + 1]) for i in range(0, 8, 2)}
    )
    circuit = Circuit([reset_layer, hadamard_layer, cx_layer])
    text = str(circuit.as_stim_circuit())
    print(hashlib.sha256(text.encode()).hexdigest())
    for line in text.splitlines():
        print(line)
    """
)


def _run_in_subprocess(pythonhashseed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=pythonhashseed)
    result = subprocess.run(
        [sys.executable, "-c", _REPRO_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


class TestStimEmissionReproducibleAcrossProcesses:
    """Regression tests exercising the fix across real process boundaries."""

    def test_full_stim_circuit_text_is_identical_across_processes(self):
        first = _run_in_subprocess("1234")
        second = _run_in_subprocess("1234")
        assert first == second

    def test_full_stim_circuit_hash_is_identical_across_processes(self):
        first_hash = _run_in_subprocess("1234").splitlines()[0]
        second_hash = _run_in_subprocess("1234").splitlines()[0]
        assert len(first_hash) == len(hashlib.sha256(b"").hexdigest())
        assert first_hash == second_hash

    def test_instruction_target_order_is_identical_across_processes(self):
        # The instruction lines (everything after the sha256 header) capture
        # both the target order within each instruction and the qubit
        # indices assigned by `default_qubit_mapping`.
        first_lines = _run_in_subprocess("5678").splitlines()[1:]
        second_lines = _run_in_subprocess("5678").splitlines()[1:]
        assert first_lines == second_lines


class TestGateHashNoLongerDependsOnClassObjectIdentity:
    """In-process checks that the hash formula itself changed as intended."""

    @pytest.mark.parametrize("gate", [H(Qubit(0)), X(Qubit(0)), I(Qubit(0))])
    def test_one_qubit_gate_hash_matches_qualname_based_formula(self, gate):
        assert hash(gate) == hash((type(gate).__qualname__, gate.qubit))

    def test_two_operand_gate_hash_matches_qualname_based_formula(self):
        gate = CX(Qubit(0), Qubit(1))
        assert hash(gate) == hash((type(gate).__qualname__, gate.control, gate.target))

    def test_equal_gates_still_have_equal_hashes(self):
        # __eq__ was not touched by this fix; equality must still imply
        # equal hashes (the Python hash/eq contract).
        assert H(Qubit(0)) == H(Qubit(0))
        assert hash(H(Qubit(0))) == hash(H(Qubit(0)))

    def test_gates_of_different_types_on_the_same_qubit_are_unequal(self):
        assert H(Qubit(0)) != X(Qubit(0))
