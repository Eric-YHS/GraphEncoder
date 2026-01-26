import os  # 处理文件路径
import io  # 用 BytesIO 在内存里读写二进制
import json  # 写入/读取 LMDB 元信息
from dataclasses import dataclass  # 方便管理配置（可选）
from typing import Optional, Dict, Any, Tuple, List  # 类型标注

import lmdb  # 用于高效存储变长数组（pos）
import numpy as np  # 数组处理
import torch  # 张量
from tqdm import tqdm  # 进度条

from rdkit import Chem  # RDKit 分子处理
from rdkit.Chem import inchi  # 生成 InChIKey（用于分子级对齐）

from ogb.utils import smiles2graph  # SMILES -> 图
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Dataset  # PCQM4Mv2


# -------------------------
# 1) 一些小工具：序列化/反序列化 pos
# -------------------------

def _np_to_bytes(arr: np.ndarray) -> bytes:
    """把 numpy 数组无 pickle 地序列化成 bytes，便于放进 LMDB。"""
    buffer = io.BytesIO()  # 内存缓冲区
    np.save(buffer, arr, allow_pickle=False)  # 保存成 .npy 格式二进制
    return buffer.getvalue()  # 取出 bytes


def _bytes_to_np(blob: bytes) -> np.ndarray:
    """从 LMDB 里读出的 bytes 反序列化回 numpy 数组。"""
    buffer = io.BytesIO(blob)  # bytes 包装成文件对象
    buffer.seek(0)  # 指针回到开头
    return np.load(buffer, allow_pickle=False)  # 读回 numpy 数组


# -------------------------
# 2) 分子级 key：InChIKey（更稳）
# -------------------------

def _mol_to_inchikey_from_smiles(smiles: str) -> Optional[str]:
    """由 SMILES 生成 InChIKey（用作分子级对齐 key）。"""
    mol = Chem.MolFromSmiles(smiles)  # 从 SMILES 生成 RDKit mol
    if mol is None:  # SMILES 不合法
        return None
    return inchi.MolToInchiKey(mol)  # 生成 InChIKey（字符串）


def _mol3d_to_inchikey(mol3d: Chem.Mol) -> Optional[str]:
    """由 SDF mol（带/不带显式H都可能）生成 InChIKey，尽量与 SMILES 侧一致。"""
    if mol3d is None:  # 解析失败的条目
        return None
    mol_no_h = Chem.RemoveHs(mol3d)  # 去显式H，避免和 SMILES 侧原子数不一致
    return inchi.MolToInchiKey(mol_no_h)  # 生成 key


# -------------------------
# 3) 原子级对齐：把 mol3d 的坐标重排到 SMILES(=2D图) 的原子顺序
# -------------------------

def _align_pos_to_smiles(
    mol3d: Chem.Mol,
    smiles: str,
    use_chirality: bool = False,
) -> Optional[np.ndarray]:
    """
    返回: pos (N,3) float32，按 SMILES mol 的原子顺序排列（即 2D 图节点顺序）。
    失败返回 None。
    """
    if mol3d is None:  # 3D mol 不存在
        return None

    mol2d = Chem.MolFromSmiles(smiles)  # 用 SMILES 生成 2D mol（基准原子顺序）
    if mol2d is None:  # SMILES 不合法
        return None

    mol3d_nh = Chem.RemoveHs(mol3d)  # 3D mol 去 H（坐标也会同步到重原子）
    mol2d_nh = Chem.RemoveHs(mol2d)  # 2D mol 去 H（与 OGB 图节点一致）

    # 先做原子数粗检（不同则无法对齐）
    if mol3d_nh.GetNumAtoms() != mol2d_nh.GetNumAtoms():
        return None

    # 用同构匹配：match[i] = mol3d 中对应 mol2d 第 i 个原子的索引
    match = mol3d_nh.GetSubstructMatch(mol2d_nh, useChirality=use_chirality)
    if not match:
        # 有时正向匹配失败，尝试反向匹配再“反转映射”
        match_rev = mol2d_nh.GetSubstructMatch(mol3d_nh, useChirality=use_chirality)
        if not match_rev:
            return None  # 双向都失败，放弃
        # match_rev[k] = mol2d 中与 mol3d 第 k 个原子对应的索引
        # 我们需要 match[i] = mol3d 中与 mol2d 第 i 个原子对应的索引
        tmp = [None] * mol2d_nh.GetNumAtoms()  # 临时数组
        for k, i in enumerate(match_rev):  # 遍历反向映射
            tmp[i] = k  # 反转
        match = tuple(tmp)  # 转回 tuple

    # 取 3D 坐标（按 mol3d_nh 的原子顺序）
    conf = mol3d_nh.GetConformer()  # conformer
    pos3d = conf.GetPositions()  # numpy (N,3)

    # 按 match 重排到 mol2d_nh（即 2D 图）原子顺序
    pos = pos3d[list(match)].astype(np.float32)  # (N,3) float32
    return pos


# -------------------------
# 4) 构建 3D 缓存（LMDB）：把 dataset idx -> pos(N,3)
# -------------------------

def _build_or_update_3d_lmdb(
    *,
    root: str,
    sdf_path: str,
    lmdb_path: str,
    mapping_mode: str,  # "auto" | "order" | "inchikey"
    use_chirality: bool,
    sanity_check_k: int,
    map_size: int,
    max_mols: Optional[int],
) -> None:
    """
    从 train.sdf 中读取 3D，做分子级+原子级对齐，把 pos 写进 LMDB。
    """
    # 读取 smiles（只读文本，不会很大）
    smiles_ds = PCQM4Mv2Dataset(root=root, only_smiles=True)  # 只加载 SMILES + 标签
    split = smiles_ds.get_idx_split()  # 取 split
    train_idx = [int(i) for i in split["train"]]  # 训练集全量 idx（长度 337万）

    # 建一个：train dataset idx -> smiles 的访问函数（避免反复写很多代码）
    def get_smiles_by_idx(idx: int) -> str:
        smi, _ = smiles_ds[idx]  # only_smiles=True 时返回 (smiles, y)
        return smi

    # 打开 SDF（流式读）
    sdf = Chem.SDMolSupplier(sdf_path, removeHs=False)  # removeHs=False 先不动，后面统一 RemoveHs

    # LMDB 环境：map_size 要足够大（这是“最大容量”，不是实际占用）
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)  # 确保目录存在
    env = lmdb.open(
        lmdb_path,  # lmdb 文件夹路径
        map_size=map_size,  # 最大容量
        subdir=True,  # lmdb_path 是目录
        readonly=False,  # 需要写入
        lock=True,  # 写入需要锁
        readahead=False,  # 训练/随机读常关掉更省内存
        meminit=False,  # 不预初始化内存页
    )

    # 如果 LMDB 已经有 meta，说明可能跑过；这里简单读取一下，避免重复工作
    with env.begin(write=False) as txn:  # 读事务
        meta_blob = txn.get(b"__meta__")  # 取元信息
    if meta_blob is not None:
        meta = json.loads(meta_blob.decode("utf-8"))  # 解析 JSON
        if meta.get("status") == "done" and max_mols is None:
            # 已经全量完成过了，就不重复构建
            env.close()
            return

    # ---------- 分子级对齐策略选择 ----------
    # order：假设 SDF 的第 i 条就是 train_idx[i]（需要验证）
    # inchikey：对每条 SDF mol 计算 InChIKey，并查回 train_idx
    # auto：先抽样验证 order，如果通过就用 order，否则用 inchikey

    def order_mapping_ok(k: int = 200) -> bool:
        """抽样检查：SDF[i] 的 key 是否与 train_idx[i] 的 key 匹配。"""
        if len(sdf) != len(train_idx):  # 条目数不一致，直接判失败
            return False
        ok = 0  # 计数匹配数
        tot = 0  # 计数有效比较数
        step = max(1, len(train_idx) // k)  # 均匀抽样步长
        for i in range(0, len(train_idx), step):  # 按步长抽样
            if tot >= k:  # 达到抽样上限
                break
            mol3d = sdf[i]  # 第 i 个 SDF 分子
            if mol3d is None:  # 跳过坏条目
                continue
            idx = train_idx[i]  # 对应假设的 dataset idx
            smi = get_smiles_by_idx(idx)  # 取 SMILES
            key_2d = _mol_to_inchikey_from_smiles(smi)  # 2D key
            key_3d = _mol3d_to_inchikey(mol3d)  # 3D key
            if key_2d is None or key_3d is None:  # key 失败就跳过
                continue
            tot += 1  # 有效比较数+1
            if key_2d == key_3d:  # key 一致则认为顺序对齐成立
                ok += 1
        # 要求匹配率足够高（比如 > 0.98）
        return (tot > 50) and (ok / tot > 0.98)

    if mapping_mode == "auto":
        mapping_mode = "order" if order_mapping_ok(k=sanity_check_k) else "inchikey"

    # inchikey 模式需要先建 train 集 key->idx 的哈希表
    inchi2idxs: Dict[str, List[int]] = {}
    if mapping_mode == "inchikey":
        for idx in tqdm(train_idx, desc="Build InChIKey->idx map (train)", total=len(train_idx)):
            smi = get_smiles_by_idx(idx)  # 取 SMILES
            key = _mol_to_inchikey_from_smiles(smi)  # 2D InChIKey
            if key is None:
                continue
            inchi2idxs.setdefault(key, []).append(idx)  # 允许重复 key（极少见）

    # ---------- 正式遍历 SDF，写入 LMDB ----------
    # 统计量：方便你知道失败原因
    n_written = 0  # 写入成功数
    n_mol_fail = 0  # 分子级匹配失败
    n_atom_fail = 0  # 原子级对齐失败

    # 写事务建议批量提交（每隔 N 条 commit 一次），速度差异巨大
    commit_every = 2048  # 每 2048 条提交一次（你可调大一点更快）
    pending = 0  # 本批未提交条数

    txn = env.begin(write=True)  # 开始写事务（长事务 + 周期性 commit）
    try:
        for i, mol3d in enumerate(tqdm(sdf, desc=f"Align 3D->2D and write LMDB ({mapping_mode})", total=len(sdf))):
            if max_mols is not None and i >= max_mols:
                break  # 只处理前 max_mols 条（用于小规模验证）

            if mol3d is None:
                n_mol_fail += 1  # 解析失败也算分子级失败
                continue

            # 1) 分子级：确定它对应 dataset 的哪个 idx
            if mapping_mode == "order":
                # 直接假设 SDF[i] 对应 train_idx[i]
                if i >= len(train_idx):
                    n_mol_fail += 1
                    continue
                idx = train_idx[i]
            elif mapping_mode == "inchikey":
                key3d = _mol3d_to_inchikey(mol3d)
                cand = inchi2idxs.get(key3d, []) if key3d is not None else []
                if len(cand) == 0:
                    n_mol_fail += 1
                    continue
                idx = cand[0]  # 多候选时先取第一个（通常不会多）
            else:
                raise ValueError(f"Unknown mapping_mode: {mapping_mode}")

            # 2) 原子级：对齐坐标到 2D 原子顺序
            smi = get_smiles_by_idx(idx)
            pos = _align_pos_to_smiles(mol3d, smi, use_chirality=use_chirality)
            if pos is None:
                n_atom_fail += 1
                continue

            # 3) 写入 LMDB：key 用 dataset idx 的字符串
            key = str(idx).encode("utf-8")
            val = _np_to_bytes(pos)  # pos (N,3) -> bytes
            txn.put(key, val)  # 写入

            n_written += 1
            pending += 1

            # 4) 定期提交，避免事务太大
            if pending >= commit_every:
                txn.commit()  # 提交本批写入
                txn = env.begin(write=True)  # 开启新事务
                pending = 0  # 清零计数

        # 循环结束后提交剩余部分
        txn.commit()

        # 写 meta（标记完成/部分完成）
        meta = {
            "status": "done" if max_mols is None else "partial",
            "mapping_mode": mapping_mode,
            "use_chirality": use_chirality,
            "n_written": n_written,
            "n_mol_fail": n_mol_fail,
            "n_atom_fail": n_atom_fail,
            "max_mols": max_mols,
        }
        with env.begin(write=True) as txn2:
            txn2.put(b"__meta__", json.dumps(meta).encode("utf-8"))

    except Exception:
        # 出错时尽量安全提交/回滚
        try:
            txn.abort()
        except Exception:
            pass
        env.close()
        raise

    env.close()  # 关闭 LMDB


# -------------------------
# 5) Dataset 包装：训练集自动带上 pos（从 LMDB 读）
# -------------------------

class PCQM4Mv2With3D(torch.utils.data.Dataset):
    """
    包装 PyG dataset：返回 Data，并在 train 时附加 data.pos。
    """
    def __init__(self, pyg_dataset: PygPCQM4Mv2Dataset, indices: np.ndarray, lmdb_path: Optional[str]):
        self.pyg_dataset = pyg_dataset  # 原始 2D PyG 数据集
        self.indices = [int(i) for i in indices]  # 该 split 的索引列表
        self.lmdb_path = lmdb_path  # LMDB 路径（valid/test 可为 None）
        self._env = None  # LMDB env（延迟打开，兼容多进程 dataloader）

    def _get_env(self):
        """延迟打开 LMDB；在每个 worker 进程里各自打开更安全。"""
        if self.lmdb_path is None:
            return None
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                subdir=True,
                readonly=True,  # 只读
                lock=False,  # 只读可以不加锁
                readahead=False,
                meminit=False,
            )
        return self._env

    def __len__(self) -> int:
        return len(self.indices)  # split 的样本数

    def __getitem__(self, i: int):
        idx = self.indices[i]  # 取全局 dataset index
        data = self.pyg_dataset[idx]  # 取 2D 图（PyG Data）

        env = self._get_env()  # 打开/获取 LMDB
        if env is not None:
            with env.begin(write=False) as txn:  # 开读事务
                blob = txn.get(str(idx).encode("utf-8"))  # 取出 pos bytes

            if blob is None:
                # 没有 3D：可以选择报错或跳过。这里选择显式报错，方便你发现缓存问题。
                raise KeyError(f"3D pos not found in LMDB for idx={idx}")
            pos = _bytes_to_np(blob)  # bytes -> numpy (N,3)
            data.pos = torch.from_numpy(pos)  # 挂到 Data 上（torch.Tensor）
        data.idx = int(idx)

        return data  # 返回 Data（train 有 pos，valid/test 没有）

def _load_or_build_train_idx_with_pos(
    *,
    pyg_len: int,
    train_idx: np.ndarray,
    lmdb_path: str,
    cache_dir: str,
    cache_name: str = "train_idx_with_pos.npy",
) -> np.ndarray:
    """
    返回过滤后的 train_idx（只保留 LMDB 中存在 pos 的 idx）。
    会把结果缓存到 cache_dir/cache_name，下次直接读取。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name)

    # 1) 如果已有缓存，直接读
    if os.path.exists(cache_path):
        arr = np.load(cache_path)
        return arr.astype(np.int64)

    # 2) 扫描 LMDB keys -> has_pos mask
    has_pos = np.zeros(pyg_len, dtype=np.bool_)  # bool mask，约 3.3M，内存很小
    env = lmdb.open(
        lmdb_path,
        subdir=True,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin(write=False) as txn:
        cur = txn.cursor()
        for k, _ in cur:
            if k == b"__meta__":
                continue
            idx = int(k.decode("utf-8"))
            if 0 <= idx < pyg_len:
                has_pos[idx] = True
    env.close()

    # 3) 过滤 train_idx
    train_idx = np.asarray(train_idx, dtype=np.int64)
    train_idx_pos = train_idx[has_pos[train_idx]]

    # 4) 写缓存
    np.save(cache_path, train_idx_pos)
    return train_idx_pos

# -------------------------
# 6) 统一入口：get_pcqm4m_dataset
# -------------------------

def get_pcqm4m_dataset(
    *,
    root: str,
    sdf_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    build_3d_cache_if_missing: bool = False,
    mapping_mode: str = "auto",
    use_chirality: bool = False,
    sanity_check_k: int = 200,
    map_size: int = 1 << 40,
    max_mols: Optional[int] = None,
    # --- SPD cache ---
    build_spd_cache_if_missing: bool = True,
    spd_max_dist: int = 8,
    spd_map_size: int = 1 << 40,
) -> Dict[str, Any]:

    pcqm_dir = os.path.join(root, "pcqm4m-v2")

    if sdf_path is None:
        candidate = os.path.join(pcqm_dir, "pcqm4m-v2-train.sdf")
        sdf_path = candidate if os.path.exists(candidate) else None

    if cache_dir is None:
        cache_dir = os.path.join(pcqm_dir, "pos_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # -------------------------
    # (A) 3D pos LMDB
    # -------------------------
    pos_lmdb_path = os.path.join(cache_dir, "pcqm4m_v2_train_pos.lmdb")

    pyg_dataset = PygPCQM4Mv2Dataset(root=root, smiles2graph=smiles2graph)
    split_idx = pyg_dataset.get_idx_split()

    data_mdb = os.path.join(pos_lmdb_path, "data.mdb")
    if sdf_path is not None:
        if (not os.path.exists(data_mdb)) and build_3d_cache_if_missing:
            _build_or_update_3d_lmdb(
                root=root,
                sdf_path=sdf_path,
                lmdb_path=pos_lmdb_path,
                mapping_mode=mapping_mode,
                use_chirality=use_chirality,
                sanity_check_k=sanity_check_k,
                map_size=map_size,
                max_mols=max_mols,
            )
        elif not os.path.exists(data_mdb):
            pos_lmdb_path = None
    else:
        pos_lmdb_path = None

    train_idx = split_idx["train"]
    filtered_train_idx = train_idx
    if pos_lmdb_path is not None:
        filtered_train_idx = _load_or_build_train_idx_with_pos(
            pyg_len=len(pyg_dataset),
            train_idx=np.asarray(train_idx),
            lmdb_path=pos_lmdb_path,
            cache_dir=cache_dir,
            cache_name="train_idx_with_pos.npy",
        )

    # ✅ 只包装 train（你实际会用它，再 random_split）
    train_ds = PCQM4Mv2With3D(pyg_dataset, filtered_train_idx, pos_lmdb_path)

    # -------------------------
    # (B) SPD LMDB（只对 filtered_train_idx 建）
    # -------------------------
    spd_lmdb_path = None
    if (pos_lmdb_path is not None) and build_spd_cache_if_missing:
        spd_lmdb_path = os.path.join(cache_dir, f"pcqm4m_v2_spd_md{spd_max_dist}.lmdb")
        spd_data_mdb = os.path.join(spd_lmdb_path, "data.mdb")

        if not os.path.exists(spd_data_mdb):
            from utils.data import build_spd_lmdb  # 你的实现位置
            # ✅ 只对真正会被训练用到的 idx 构建（filtered_train_idx）
            idx_list = [int(i) for i in filtered_train_idx]
            build_spd_lmdb(
                pyg_dataset=pyg_dataset,
                indices=idx_list,
                lmdb_path=spd_lmdb_path,
                spd_max_dist=int(spd_max_dist),
                map_size=int(spd_map_size),
                max_mols=max_mols,   # 你调试时可用
            )
    elif (pos_lmdb_path is not None) and (not build_spd_cache_if_missing):
        # 不构建时，如果你仍然希望走 SPD，就要求它已存在
        cand = os.path.join(cache_dir, f"pcqm4m_v2_spd_md{spd_max_dist}.lmdb")
        if os.path.exists(os.path.join(cand, "data.mdb")):
            spd_lmdb_path = cand
        else:
            spd_lmdb_path = None

    # （可选）valid/test 你可以仍然返回，但你说你不使用，完全可以设为 None 省点心
    valid_key = "valid" if "valid" in split_idx else "val"
    valid_ds = PCQM4Mv2With3D(pyg_dataset, split_idx[valid_key], None)
    test_ds = PCQM4Mv2With3D(pyg_dataset, split_idx.get("test", []), None) if "test" in split_idx else None

    return {
        "pyg_dataset": pyg_dataset,
        "split_idx": split_idx,

        "train": train_ds,
        "valid": valid_ds,
        "test": test_ds,

        "lmdb_path": pos_lmdb_path,
        "sdf_path": sdf_path,
        "cache_dir": cache_dir,
        "train_idx_with_pos_path": os.path.join(cache_dir, "train_idx_with_pos.npy") if pos_lmdb_path is not None else None,
        "train_size_before_filter": int(len(train_idx)),
        "train_size_after_filter": int(len(filtered_train_idx)),

        # ✅ NEW: SPD cache info
        "spd_lmdb_path": spd_lmdb_path,
        "spd_max_dist": int(spd_max_dist),
    }
