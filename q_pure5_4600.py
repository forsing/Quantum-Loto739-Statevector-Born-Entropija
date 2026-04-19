#!/usr/bin/env python3
"""
Pet čisto kvantnih modela:
ceo učitani CSV → mean, std, median, min, max po 7 kolona 
→ topologije Ry/CNOT, cirk. CNOT, amp+H^3, Ry+RZZ, amp+QFT.
Predikcija 4601: 
leks. indeks C(39,7) od imax, <Z_i>, indeks modela 5..9, digest celog H (bez ML treninga).
"""

from __future__ import annotations

import csv
import warnings
from math import comb
from pathlib import Path
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")


def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(7)])
    return np.array(rows, dtype=int)


def row_thetas(row: np.ndarray) -> np.ndarray:
    """Ugao po komponenti u (0, pi) iz brojeva 1..39."""
    r = row.astype(np.float64)
    return (r - 1.0) / 38.0 * np.pi


def vec_to_thetas(v: np.ndarray) -> np.ndarray:
    """Skalira proizvoljan 7-vektor iz CSV agregata u [0, π] po kolonama (fiksno, nije ML)."""
    v = np.asarray(v, dtype=np.float64)
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-15:
        return np.full(7, np.pi / 2.0, dtype=np.float64)
    u = (v - lo) / (hi - lo)
    return u * np.pi


def digest_whole_H(H: np.ndarray) -> int:
    """Kompaktan ceo-niz int (fiksno) za readout, bez ML."""
    s = int(np.sum(H, dtype=np.int64))
    return int((s * 1_009_741 + H.shape[0] * 50_027 + H.shape[1] * 3_019) & 0x7FFFFFFF)


def summary_vec_csv(agg_idx: int, H: np.ndarray) -> np.ndarray:
    """Sedam brojeva po kolonama iz CELOG H — samo M6–M10; agg_idx 0=mean … 4=max."""
    X = H.astype(np.float64)
    if agg_idx == 0:
        return X.mean(axis=0)
    if agg_idx == 1:
        return X.std(axis=0)
    if agg_idx == 2:
        return np.median(X, axis=0)
    if agg_idx == 3:
        return X.min(axis=0)
    return X.max(axis=0)


def _born_probs(sv: Statevector) -> np.ndarray:
    return np.abs(sv.data) ** 2


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 1e-18]
    return float(-np.sum(p * np.log2(p)))


def unrank_combo_1based(n: int, k: int, index: int) -> Tuple[int, ...]:
    """Leksikografski index-ta rastuća k-torka iz {1..n} (0-based index)."""
    tot = comb(n, k)
    idx = int(index) % tot
    result: List[int] = []
    a = 1
    for i in range(k):
        for j in range(a, n + 1):
            c = comb(n - j, k - i - 1)
            if idx >= c:
                idx -= c
            else:
                result.append(j)
                a = j + 1
                break
    return tuple(result)


def pred_4601_from_quantum(ez: List[float], imax: int, model_idx: int, digest: int) -> Tuple[int, ...]:
    """Fiksno: indeks u C(39,7) od imax, model_idx, <Z> i digest celog CSV (nije ML)."""
    z = np.asarray(ez, dtype=np.float64)
    s = int(np.sum(np.abs(z) * 1e6) + np.sum(z * 1e5))
    rot = int(np.sum(z * np.array([1, 3, 5, 7, 11, 13, 17][: z.size], dtype=np.float64)))
    h = imax * 1_048_579 + model_idx * 79_199 + s + rot * 193 + (digest & 0x7FFFFFFF)
    h ^= h >> 16
    h = (h * 0x9E3779B9) & 0x7FFFFFFF
    return unrank_combo_1based(39, 7, h)


def _z_expectations(n: int, probs: np.ndarray) -> List[float]:
    out: List[float] = []
    for i in range(n):
        e = 0.0
        for idx in range(1 << n):
            bit = (idx >> i) & 1
            e += probs[idx] * ((-1.0) ** bit)
        out.append(float(e))
    return out


def _print_model(name: str, qc: QuantumCircuit, model_idx: int, digest: int) -> None:
    sv = Statevector(qc)
    n = qc.num_qubits
    probs = _born_probs(sv)
    ent = _entropy_bits(probs)
    pmax = float(np.max(probs))
    imax = int(np.argmax(probs))
    ez = _z_expectations(n, probs)
    pred = pred_4601_from_quantum(ez, imax, model_idx, digest)
    print(name)
    print("  qubit-a:", n, "| entropija (bit):", round(ent, 6), "| max p:", round(pmax, 6), "| argmax indeks:", imax)
    print("  <Z_i>:", tuple(round(z, 5) for z in ez))
    print("  predikcija 4601:", pred)


# --- M1: Ry uglovi + linearni CNOT lanac ---
def build_m1(theta: np.ndarray) -> QuantumCircuit:
    n = 7
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(theta[i]), i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


# --- M2: Ry + cirkularni CNOT (i -> (i+1) mod n) ---
def build_m2(theta: np.ndarray) -> QuantumCircuit:
    n = 7
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(theta[i]), i)
    for i in range(n):
        qc.cx(i, (i + 1) % n)
    return qc


# --- M3: 3 qubita, amplitude iz 8 vektora (7 brojeva + 0), zatim H^⊗3 ---
def build_m3(row7: np.ndarray) -> QuantumCircuit:
    v = np.zeros(8, dtype=np.complex128)
    r = row7.astype(np.float64)
    v[:7] = r - np.mean(r)
    nv = float(np.linalg.norm(v)) or 1.0
    v = v / nv
    qc = QuantumCircuit(3)
    qc.initialize(v)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    return qc


# --- M4: Ry + RZZ parovi (uglovi iz proizvoda susednih komponenti) ---
def build_m4(theta: np.ndarray) -> QuantumCircuit:
    n = 7
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(theta[i]), i)
    for i in range(n - 1):
        g = float(theta[i] * theta[i + 1])
        qc.rzz(2.0 * g, i, i + 1)
    return qc


# --- M5: 3 qubita amplitude + QFT ---
def build_m5(row7: np.ndarray) -> QuantumCircuit:
    v = np.zeros(8, dtype=np.complex128)
    r = row7.astype(np.float64)
    v[:7] = r
    nv = float(np.linalg.norm(v)) or 1.0
    v = v / nv
    qc = QuantumCircuit(3)
    qc.initialize(v)
    try:
        from qiskit.circuit.library import QFTGate

        qc.append(QFTGate(3), [0, 1, 2])
    except ImportError:
        from qiskit.circuit.library import QFT

        qc.append(QFT(3, do_swaps=False), [0, 1, 2])
    return qc


def main() -> int:
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*basis_change\.qft\.QFT.*")

    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1
    dig = digest_whole_H(H)
    print("CSV:", CSV_PATH)
    print(
        "q_pure5 (pet modela u ovom fajlu): redova iz CSV =",
        H.shape[0],
        "| agregacija mean..max po svim redovima | digest(celo H):",
        dig,
    )
    print("---")

    specs: List[Tuple[str, QuantumCircuit, int]] = []
    for j in range(5):
        v7 = summary_vec_csv(j, H)
        th = vec_to_thetas(v7)
        mid = 5 + j
        if j == 0:
            specs.append(("M6 mean(ce CSV)→Ry + CNOT lanac", build_m1(th), mid))
        elif j == 1:
            specs.append(("M7 std(ce CSV)→Ry + cirk. CNOT", build_m2(th), mid))
        elif j == 2:
            specs.append(("M8 median(ce CSV)→amp + H^3", build_m3(v7 - np.mean(v7)), mid))
        elif j == 3:
            specs.append(("M9 min(ce CSV)→Ry + RZZ", build_m4(th), mid))
        else:
            specs.append(("M10 max(ce CSV)→amp + QFT", build_m5(v7), mid))

    for i, (name, qc, mid) in enumerate(specs):
        _print_model(name, qc, mid, dig)
        if i < len(specs) - 1:
            print("---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
CSV: /data/loto7hh_4600_k31.csv
q_pure5 (pet modela u ovom fajlu): redova iz CSV = 4600 | agregacija mean..max po svim redovima | digest(celo H): 50299360
---
M6 mean(ce CSV)→Ry + CNOT lanac
  qubit-a: 7 | entropija (bit): 3.312762 | max p: 0.24824 | argmax indeks: 80
  <Z_i>: (1.0, 0.87114, 0.44445, 0.00246, -0.00122, 0.00106, -0.00106)
  predikcija 4601: (10, 18, 21, 22, 24, 27, 29)
---
M7 std(ce CSV)→Ry + cirk. CNOT
  qubit-a: 7 | entropija (bit): 2.285896 | max p: 0.394962 | argmax indeks: 107
  <Z_i>: (-0.07749, -0.33696, 0.32508, -0.32508, 0.30086, -0.07637, -0.07637)
  predikcija 4601: (3, 5, 6, 20, 25, 26, 36)
---
M8 median(ce CSV)→amp + H^3
  qubit-a: 3 | entropija (bit): 1.787174 | max p: 0.636816 | argmax indeks: 4
  <Z_i>: (0.57463, 0.39801, -0.699)
  predikcija 4601: (3, 6, 16, 21, 27, 30, 33)
---
M9 min(ce CSV)→Ry + RZZ
  qubit-a: 7 | entropija (bit): 3.063383 | max p: 0.301674 | argmax indeks: 96
  <Z_i>: (1.0, 0.95949, 0.84125, 0.41542, 0.14231, -0.65486, -1.0)
  predikcija 4601: (2, 6, 17, 21, 24, 30, 33)
---
M10 max(ce CSV)→amp + QFT
  qubit-a: 3 | entropija (bit): 0.940343 | max p: 0.865953 | argmax indeks: 0
  <Z_i>: (0.83267, 0.84536, 0.85106)
  predikcija 4601: (16, 18, 20, 21, 24, 32, 37)
"""



"""
Pet odvojenih „čisto kvantnih“ scenarija 
gde ceo istorijski CSV ulazi samo kao sedam brojeva po koloni 
(pet statistika x sedam kolona), 
a izlazna sedmorka je deterministički izvedena iz statistike stanja 
i digest-a, bez učenja iz parova kao u q_scan.

Učitava ceo CSV kao matricu H (sedmorke). 
Iz svih redova pravi pet sedmorki agregata po kolonama: 
mean, std, median, min, max (summary_vec_csv). 
Svaka od tih sedmorki ulazi u jedno fiksno kvantno kolo 
(bez učenja parametara iz podataka — nema ML treninga). 
Računa se tačno Stanje (Statevector), 
zatim Born verovatnoće, entropija, max verovatnoća, indeks maksimuma i ⟨Zᵢ⟩. 
„Predikcija 4601“ nije regresija: 
deterministička funkcija pred_4601_from_quantum mapira 
imax, ⟨Z⟩, indeks modela 5…9 i digest_whole_H(H) 
(hash celog niza) u jednu rastuću sedmorku iz C(39,7) preko unrank_combo_1based.

Metode i tehnike (M6-M10 u štampi; 
graditelji u kodu build_m1…build_m5)

Model	Ulaz iz CSV-a	Kolo
M6
mean po 7 kolona → vec_to_thetas
Ry na 7 kubit-a + linearni CNOT lanac
M7
std → uglovi
Ry + cirkularni CNOT (i → (i+1) mod 7)
M8
median (centrirano za amplitudu)
3 qubita: initialize na 8-dim vektoru (7 brojeva + 0), zatim H⊗H⊗H
M9
min → uglovi
Ry + RZZ na susedima (ugao ∝ θᵢθᵢ₊₁)
M10
max
3 qubita: normalizovana amplituda iz 7 vrednosti + QFT
digest_whole_H i model_idx razdvajaju readout između modela i vezuju ga za ceo niz.

Prednosti
Ceo CSV ulazi u statistiku (mean…max po svim redovima) i u digest — nema „samo poslednji red“ za agregat.
Bez treninga: reproduktibilno ako su kolo i CSV isti; nema overfittinga u ML smislu.
Pet različitih topologija → različita stanja, entropije, ⟨Z⟩ → različite (ili bar odvojive) sedmorke u readout-u.
Lokalno: Statevector + mali broj kubit-a (7 ili 3).

Nedostaci
Predikcija nije model sledećeg izvlačenja — to je fiksna heuristika (miks bitova + unrank); nema teorijske veze sa loto procesom.
⟨Zᵢ⟩ za n=7 računa se petljom preko 2ⁿ stanja u Pythonu — skalabilnost loša (za 7 još uvek OK).
M3/M10 koriste samo 7 brojeva kao vektor vrednosti (gubi se permutacija unutar sedmorke u tom koraku) — grublji signal od Ry-topologija na 7 kubit-a.
Agregati (posebno std/min/max) mogu biti osetljivi na outlier-e ili na oblik raspodele; min/max posebno „hvataju“ ekstreme celog istorijskog niza.
"""
