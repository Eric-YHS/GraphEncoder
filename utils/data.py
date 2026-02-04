import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from collections import deque
import numpy as np
import torch

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}


class PDBProtein(object):
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass

        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.long),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=np.bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.long)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.long),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand['pos']:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block


def parse_pdbbind_index_file(path):
    pdb_id = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        pdb_id.append(line.split()[0])
    return pdb_id


def parse_sdf_file(path):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # read mol
    if path.endswith('.sdf'):
        rdmol = Chem.MolFromMolFile(path, sanitize=False)
    elif path.endswith('.mol2'):
        rdmol = Chem.MolFromMol2File(path, sanitize=False)
    else:
        raise ValueError
    Chem.SanitizeMol(rdmol)
    rdmol = Chem.RemoveHs(rdmol)

    # Remove Hydrogens.
    # rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True)))
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    # Get hybridization in the order of atom idx.
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    ptable = Chem.GetPeriodicTable()

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int)

    # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.long)
    edge_type = np.array(edge_type, dtype=np.long)

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization
    }
    return data


# utils/spd_cache.py
import os
import lmdb
import numpy as np
import torch
from collections import deque
from typing import Iterable, Optional

# ========== SPD compute (BFS) ==========
def compute_spd_matrix(edge_index: torch.Tensor, num_nodes: int, max_dist: int) -> np.ndarray:
    """
    返回 uint8 的 SPD 矩阵 [n,n]，值在 [0, max_dist+1]，不可达=max_dist+1。
    edge_index: [2,E] (局部 0..n-1)
    """
    INF = max_dist + 1

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)
            adj[v].append(u)  # 分子图视为无向

    dist = np.full((num_nodes, num_nodes), INF, dtype=np.uint8)
    for s in range(num_nodes):
        dist[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = int(dist[s, u])
            if du >= INF:
                continue
            nd = du + 1
            if nd > INF:
                continue
            for v in adj[u]:
                if dist[s, v] > nd:
                    dist[s, v] = nd
                    q.append(v)
    return dist


def compute_spd_and_edge_input(
    edge_index: torch.Tensor,   # [2,E]
    edge_attr: torch.Tensor,    # [E,3]
    num_nodes: int,
    max_dist: int,
    bond_dims,
    num_edge_types: int,
):
    """
    返回：
      spd: uint8 [n,n]，值域 [0, max_dist+1]，>max_dist 视作 max_dist+1
      edge_input: uint16 [n,n,max_dist]，每跳的 edge_type_id（0=pad）
    """
    n = num_nodes
    INF = max_dist + 1

    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()

    edge_type = pack_edge_type(edge_attr, bond_dims).cpu().tolist()  # [E] in 1..num_edge_types

    # 邻接表：无向处理（如果 edge_index 本身已有双向，重复也没事）
    adj = [[] for _ in range(n)]
    for u, v, et in zip(src, dst, edge_type):
        if 0 <= u < n and 0 <= v < n:
            adj[u].append((v, et))
            adj[v].append((u, et))

    spd = np.full((n, n), INF, dtype=np.uint8)
    edge_input = np.zeros((n, n, max_dist), dtype=np.uint16)

    for s in range(n):
        # BFS arrays
        dist = np.full(n, INF, dtype=np.int16)
        pred_node = np.full(n, -1, dtype=np.int16)
        pred_edge = np.zeros(n, dtype=np.uint16)

        dist[s] = 0
        q = deque([s])

        while q:
            u = q.popleft()
            du = int(dist[u])
            if du >= max_dist:     # ✅ 关键：超过 max_dist 不再扩展
                continue
            nd = du + 1
            for v, et in adj[u]:
                if dist[v] > nd:
                    dist[v] = nd
                    pred_node[v] = u
                    pred_edge[v] = et
                    q.append(v)

        # 写 SPD 行
        # dist>max_dist 的仍然保持 INF
        dist_clamped = np.minimum(dist, INF).astype(np.uint8)
        spd[s, :] = dist_clamped
        spd[s, s] = 0

        # 写 edge_input[s, t, :]
        for t in range(n):
            dt = int(dist[t])
            if t == s or dt == INF or dt == 0 or dt > max_dist:
                continue

            # 回溯 t -> s 收集边类型
            edges = []
            cur = t
            steps = 0
            while cur != s and cur != -1 and steps < max_dist:
                edges.append(int(pred_edge[cur]))
                cur = int(pred_node[cur])
                steps += 1

            if cur != s:
                # 理论上不会发生（除非图里异常或 pred 断裂）
                continue

            edges.reverse()  # hop 顺序：从 s 出发
            L = min(len(edges), max_dist)
            if L > 0:
                edge_input[s, t, :L] = np.asarray(edges[:L], dtype=np.uint16)

    return spd, edge_input

# ========== LMDB pack/unpack ==========
def pack_edge_type(edge_attr: torch.Tensor, bond_dims) -> torch.Tensor:
    """
    edge_attr: [E,3]，每列是离散 id
    返回 edge_type_id: [E]，范围 1..num_edge_types，0 留给 padding
    """
    d0, d1, d2 = bond_dims
    a0 = edge_attr[:, 0].long()
    a1 = edge_attr[:, 1].long()
    a2 = edge_attr[:, 2].long()
    edge_type = a0 + d0 * (a1 + d1 * a2)  # 0..num_edge_types-1
    return edge_type + 1                  # 1..num_edge_types

def spd_pack(spd_uint8: np.ndarray) -> bytes:
    """
    format:
      uint16 n  (2 bytes, little endian)
      uint8  spd_flat (n*n bytes)
    """
    n = spd_uint8.shape[0]
    assert spd_uint8.shape == (n, n)
    header = np.array([n], dtype=np.uint16).tobytes()
    payload = spd_uint8.reshape(-1).tobytes()
    return header + payload

def spd_unpack(blob: bytes) -> np.ndarray:
    n = np.frombuffer(blob[:2], dtype=np.uint16)[0].item()
    arr = np.frombuffer(blob[2:], dtype=np.uint8)
    return arr.reshape((n, n))

def spd_edge_pack(spd_uint8: np.ndarray, edge_input_u16: np.ndarray, max_dist: int, num_edge_types: int) -> bytes:
    """
    format (little endian):
      uint16 n
      uint16 max_dist
      uint32 num_edge_types
      uint8  spd_flat (n*n bytes)
      uint16 edge_input_flat (n*n*max_dist*2 bytes)
    """
    n = spd_uint8.shape[0]
    assert spd_uint8.shape == (n, n)
    assert edge_input_u16.shape == (n, n, max_dist)

    header = (
        np.array([n], dtype=np.uint16).tobytes() +
        np.array([max_dist], dtype=np.uint16).tobytes() +
        np.array([num_edge_types], dtype=np.uint32).tobytes()
    )
    payload_spd = spd_uint8.reshape(-1).tobytes()
    payload_edge = edge_input_u16.reshape(-1).tobytes()
    return header + payload_spd + payload_edge

def spd_edge_unpack(blob: bytes):
    n = np.frombuffer(blob[:2], dtype=np.uint16)[0].item()
    max_dist = np.frombuffer(blob[2:4], dtype=np.uint16)[0].item()
    num_edge_types = np.frombuffer(blob[4:8], dtype=np.uint32)[0].item()

    spd_bytes = n * n
    spd_start = 8
    spd_end = spd_start + spd_bytes
    spd = np.frombuffer(blob[spd_start:spd_end], dtype=np.uint8).reshape((n, n))

    edge = np.frombuffer(blob[spd_end:], dtype=np.uint16).reshape((n, n, max_dist))
    return spd, edge, max_dist, num_edge_types


# ========== Builder ==========
def build_spd_edge_lmdb(
    pyg_dataset,
    indices: Iterable[int],
    lmdb_path: str,
    spd_max_dist: int,
    map_size: int = (1 << 40),
    max_mols: Optional[int] = None,
    verbose_every: int = 10000,
):
    """
    读取 pyg_dataset[idx] 的 edge_index / edge_attr / num_nodes，
    计算：
      - SPD: uint8 [n,n] in [0, max_dist+1]，>max_dist 记为 max_dist+1
      - edge_input: uint16 [n,n,max_dist]，每跳的 edge_type_id（0=pad）
    并写入 LMDB。
    key = str(idx)
    """
    os.makedirs(lmdb_path, exist_ok=True)
    data_mdb = os.path.join(lmdb_path, "data.mdb")
    if os.path.exists(data_mdb):
        print(f"[SPD+EDGE-LMDB] exists: {data_mdb}, skip build.")
        return

    # ✅ 用 OGB 的 bond feature dims 来做稳定的 edge_type 打包（推荐）
    try:
        from ogb.utils.features import get_bond_feature_dims
        bond_dims = get_bond_feature_dims()  # len=3
        if len(bond_dims) != 3:
            raise ValueError(f"Expected 3 bond dims, got {bond_dims}")
        d0, d1, d2 = [int(x) for x in bond_dims]
        num_edge_types = d0 * d1 * d2
    except Exception as e:
        raise RuntimeError(
            "Cannot import ogb.utils.features.get_bond_feature_dims(). "
            "Please ensure ogb is installed, or provide a fixed bond_dims."
        ) from e

    env = lmdb.open(
        lmdb_path,
        subdir=True,
        map_size=map_size,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        max_dbs=1,
    )

    count = 0
    with env.begin(write=True) as txn:
        for k, idx in enumerate(indices):
            if max_mols is not None and count >= max_mols:
                break

            data = pyg_dataset[int(idx)]
            n = int(data.num_nodes)

            ei = data.edge_index
            ea = data.edge_attr
            if not torch.is_tensor(ei):
                ei = torch.as_tensor(ei, dtype=torch.long)
            else:
                ei = ei.long()
            if not torch.is_tensor(ea):
                ea = torch.as_tensor(ea, dtype=torch.long)
            else:
                ea = ea.long()

            if ea.size(-1) != 3:
                raise ValueError(f"edge_attr last dim must be 3, got {ea.shape} at idx={idx}")

            spd, edge_input = compute_spd_and_edge_input(
                edge_index=ei,
                edge_attr=ea,
                num_nodes=n,
                max_dist=int(spd_max_dist),
                bond_dims=(d0, d1, d2),
                num_edge_types=num_edge_types,
            )

            blob = spd_edge_pack(spd, edge_input, max_dist=int(spd_max_dist), num_edge_types=num_edge_types)
            txn.put(str(int(idx)).encode("utf-8"), blob)

            count += 1
            if verbose_every and (count % verbose_every == 0):
                print(f"[SPD+EDGE-LMDB] built {count} molecules...")

    env.sync()
    env.close()
    print(f"[SPD+EDGE-LMDB] done. total={count} saved at {lmdb_path}")


def build_spd_lmdb(
    pyg_dataset,
    indices: Iterable[int],
    lmdb_path: str,
    spd_max_dist: int,
    map_size: int = (1 << 40),
    max_mols: Optional[int] = None,
    verbose_every: int = 10000,
):
    """
    读取 pyg_dataset[idx] 的 edge_index / num_nodes，计算 SPD 并写入 LMDB。
    key = str(idx)
    """
    os.makedirs(lmdb_path, exist_ok=True)
    data_mdb = os.path.join(lmdb_path, "data.mdb")
    if os.path.exists(data_mdb):
        print(f"[SPD-LMDB] exists: {data_mdb}, skip build.")
        return

    env = lmdb.open(
        lmdb_path,
        subdir=True,
        map_size=map_size,
        readonly=False,
        lock=True,          # 写入需要 lock
        readahead=False,
        meminit=False,
        max_dbs=1,
    )

    count = 0
    with env.begin(write=True) as txn:
        for k, idx in enumerate(indices):
            if max_mols is not None and count >= max_mols:
                break

            data = pyg_dataset[int(idx)]
            n = int(data.num_nodes)
            ei = data.edge_index  # [2,E] 全局图内本来就是局部索引
            if not torch.is_tensor(ei):
                ei = torch.as_tensor(ei, dtype=torch.long)

            spd = compute_spd_matrix(ei, n, spd_max_dist)  # uint8 [n,n]
            blob = spd_pack(spd)
            txn.put(str(int(idx)).encode("utf-8"), blob)

            count += 1
            if verbose_every and (count % verbose_every == 0):
                print(f"[SPD-LMDB] built {count} molecules...")

    env.sync()
    env.close()
    print(f"[SPD-LMDB] done. total={count} saved at {lmdb_path}")

import lmdb
import torch
import numpy as np
from torch_geometric.data import Batch

class CollateWithSPDLmdb:
    def __init__(self, spd_lmdb_path: str, spd_max_dist: int):
        self.spd_lmdb_path = spd_lmdb_path
        self.spd_max_dist = int(spd_max_dist)
        self._env = None

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.spd_lmdb_path,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_dbs=1,
            )
        return self._env

    def __call__(self, data_list):
        # 先做普通 Batch
        batch = Batch.from_data_list(data_list)

        # 再构造 padded SPD dense
        sizes = [int(d.num_nodes) for d in data_list]
        B = len(sizes)
        L = max(sizes) if B > 0 else 0
        INF = self.spd_max_dist + 1

        spd_dense = torch.full((B, L, L), INF, dtype=torch.long)

        env = self._get_env()
        with env.begin(write=False) as txn:
            for i, d in enumerate(data_list):
                if not hasattr(d, "idx"):
                    raise ValueError("Data object missing .idx (global dataset index). Please set data.idx in Dataset.__getitem__.")
                idx = int(d.idx)
                blob = txn.get(str(idx).encode("utf-8"))
                if blob is None:
                    raise KeyError(f"SPD not found in LMDB for idx={idx}. Did you build the SPD cache?")

                spd = spd_unpack(blob)        # uint8 [n,n]
                n = spd.shape[0]
                spd_dense[i, :n, :n] = torch.from_numpy(spd.astype(np.int64))

        batch.spatial_pos_dense = spd_dense
        return batch


class CollateWithSPDEdgeLmdb:
    """
    从 LMDB 读取：
      - spd: uint8 [n,n] in [0, spd_max_dist+1]
      - edge_input: uint16 [n,n,spd_max_dist] (0=pad, 1..num_edge_types)
    然后 pad 成 batch dense：
      - batch.spatial_pos_dense: [B,L,L] long
      - batch.edge_input_dense:  [B,L,L,spd_max_dist] long
    """
    def __init__(self, spd_lmdb_path: str, spd_max_dist: int):
        self.spd_lmdb_path = spd_lmdb_path
        self.spd_max_dist = int(spd_max_dist)
        self._env = None

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.spd_lmdb_path,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_dbs=1,
            )
        return self._env

    def __call__(self, data_list):
        batch = Batch.from_data_list(data_list)

        sizes = [int(d.num_nodes) for d in data_list]
        B = len(sizes)
        L = max(sizes) if B > 0 else 0
        INF = self.spd_max_dist + 1

        # pad SPD
        spd_dense = torch.full((B, L, L), INF, dtype=torch.long)

        # pad edge_input
        edge_input_dense = torch.zeros((B, L, L, self.spd_max_dist), dtype=torch.long)

        env = self._get_env()
        with env.begin(write=False) as txn:
            for i, d in enumerate(data_list):
                if not hasattr(d, "idx"):
                    raise ValueError(
                        "Data object missing .idx (global dataset index). "
                        "Please set data.idx in Dataset.__getitem__."
                    )
                idx = int(d.idx)
                blob = txn.get(str(idx).encode("utf-8"))
                if blob is None:
                    raise KeyError(f"SPD+EDGE not found in LMDB for idx={idx}. Did you build the SPD+EDGE cache?")

                # ✅ 新 unpack：同时读 spd 和 edge_input
                spd, edge_in, md, _ = spd_edge_unpack(blob)

                if int(md) != self.spd_max_dist:
                    raise ValueError(
                        f"LMDB max_dist={md} != collate spd_max_dist={self.spd_max_dist}. "
                        "Please rebuild cache or align config."
                    )

                n = spd.shape[0]
                spd_dense[i, :n, :n] = torch.from_numpy(spd.astype(np.int64))
                edge_input_dense[i, :n, :n, :] = torch.from_numpy(edge_in.astype(np.int64))

        batch.spatial_pos_dense = spd_dense
        batch.edge_input_dense = edge_input_dense
        return batch
