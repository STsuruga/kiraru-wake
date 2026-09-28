"""きらるわけ用の分子座標を生成する。
RDKit で 3D 埋め込み → キラリティー判定 → 必要なら鏡映して「右の箱(R/Ra/Rp/P)」側に揃える → 画面向きに回転。
出力の座標は画面系 (x 右, y 下, z 手前) で、S 側は実行時に x を反転する。
"""
import json, math, sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem import rdCIPLabeler

PX = 9.0  # px / Å

# map番号: 9=顔の環/原子, 8=2つ目の顔, 1/2=軸の手前/奥原子, 3/4=手前/奥の優先1置換基原子
# 5=面性: 橋頭(Br の隣), 6=C-Br, 7=その隣, 10=パイロット原子
MOLS = {
 'chbrclf':  dict(cat='center', smi='F[C@H:9](Cl)Br'),
 'lactic':   dict(cat='center', smi='C[C@@H:9](O)C(=O)O'),
 'alanine':  dict(cat='center', smi='C[C@@H:9](N)C(=O)O'),
 'glyceraldehyde': dict(cat='center', smi='O=C[C@H:9](O)CO'),
 'butanol':  dict(cat='center', smi='CC[C@@H:9](C)O'),
 'ibuprofen':dict(cat='center', smi='CC(C)Cc1c[c:9]cc(c1)[C@@H](C)C(=O)O'),
 'thalidomide':dict(cat='center', smi='O=C1CC[C@@H](N2C(=O)c3cc[c:9]cc3C2=O)C(=O)N1'),
 'sulfoxide':dict(cat='center', smi='C[S@@](=O)c1c[c:9]c(C)cc1'),
 'pamp':     dict(cat='center', smi='C[P@](c1cc[c:9]cc1)c1ccccc1OC'),
 'allene':   dict(cat='axial',  smi='[CH3:3][CH:1]=[C:9]=[CH:2][CH3:4]'),
 'binol':    dict(cat='axial',  smi='O[c:3]1ccc2cc[c:9]cc2[c:1]1-[c:2]1[c:4](O)ccc2ccccc12', phi=18),
 'binap':    dict(cat='axial',  smi='[PH2][c:3]1ccc2cc[c:9]cc2[c:1]1-[c:2]1[c:4]([PH2])ccc2ccccc12', phi=18),
 'diphenic': dict(cat='axial',  smi='OC(=O)c1c[c:9]c[c:3]([N+](=O)[O-])[c:1]1-[c:2]1[c:4]([N+](=O)[O-])cccc1C(=O)O', phi=18),
 'helicene': dict(cat='helix',  smi=None, rot=[([1, 0, 0], 12)]),
 'pcp':      dict(cat='planar', smi='Br[c:6]1[cH:7][c:9]2cc[c:5]1[CH2:10]Cc1ccc(cc1)CC2'),
 'tartaric': dict(cat='meso',   smi='O=C(O)[C@@H:9](O)[C@@H:8](O)C(=O)O', syn=('O','O')),
 'dibromobutane': dict(cat='meso', smi='C[C@@H:9](Br)[C@@H:8](Br)C', syn=('Br','Br')),
 'butanediol': dict(cat='meso', smi='C[C@@H:9](O)[C@@H:8](O)C', syn=('O','O')),
}
# 置換基の略記(フェニル環を描かずラベルにする)
LABEL_OVERRIDE = {'binap': {'P': 'PPh₂'}, 'diphenic': {'N': 'NO₂'}}

SUB = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')


def mapidx(m, n):
    for a in m.GetAtoms():
        if a.GetAtomMapNum() == n:
            return a.GetIdx()
    return None


def det_chir(a, b, c, d):
    """a,b,c = 優先1,2,3、d = 最低順位。右手系で < 0 なら R(時計回り)"""
    return float(np.dot(a - d, np.cross(b - d, c - d)))


def torsion(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return math.degrees(math.atan2(y, x))


def rot(axis, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, y, z = axis
    return np.array([[c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
                     [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
                     [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]])


def frame(xdir, ydir):
    """右手系の回転行列(行 = 新しい x, y, z 軸)"""
    x = xdir / np.linalg.norm(xdir)
    y = ydir - np.dot(ydir, x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.array([x, y, z])


def cip_center(m, P, idx):
    """RDKit の CIP ラベラーで R/S。"""
    mm = Chem.Mol(m)
    conf = mm.GetConformer()
    for i, p in enumerate(P):
        conf.SetAtomPosition(i, p.tolist())
    Chem.AssignStereochemistryFrom3D(mm)
    rdCIPLabeler.AssignCIPLabels(mm)
    a = mm.GetAtomWithIdx(idx)
    return a.GetProp('_CIPCode') if a.HasProp('_CIPCode') else '?'


def det_center(m, P, c):
    """CIP 順位(グラフから)+ 3D 座標の行列式で R/S。孤立電子対は中心から反対方向の擬原子。"""
    mm = Chem.Mol(m)
    Chem.AssignStereochemistry(mm, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    nb = sorted(mm.GetAtomWithIdx(c).GetNeighbors(), key=lambda n: -int(n.GetProp('_CIPRank')))
    pts = [P[n.GetIdx()] for n in nb]
    if len(pts) == 3:
        v = P[c] - np.mean(pts, axis=0)
        pts.append(P[c] + v / np.linalg.norm(v))
    return 'R' if det_chir(*pts) < 0 else 'S'


def ring_centroids_path(m, P):
    ri = m.GetRingInfo()
    rings = [list(r) for r in ri.AtomRings()]
    adj = {i: [] for i in range(len(rings))}
    for i in range(len(rings)):
        for j in range(i+1, len(rings)):
            if len(set(rings[i]) & set(rings[j])) == 2:
                adj[i].append(j); adj[j].append(i)
    ends = [i for i in adj if len(adj[i]) == 1]
    path, prev, cur = [ends[0]], None, ends[0]
    while True:
        nxt = [k for k in adj[cur] if k != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        path.append(cur)
    return [P[rings[k]].mean(axis=0) for k in path], [rings[k] for k in path]


def descriptor(key, spec, m, P):
    """右の箱側なら True"""
    cat = spec['cat']
    if cat == 'center':
        c = spec['_center']
        code = cip_center(m, P, c)
        det_code = det_center(m, P, c)
        if code == '?':
            code = det_code  # RDKit が 3D から付けられない(リン中心など)ときは行列式で判定
        assert code == det_code, (key, code, det_code)  # 2つの方法が一致することを確認
        spec['_code'] = code
        return code == 'R'
    if cat == 'axial':
        f, r, f1, r1 = (mapidx(m, k) for k in (1, 2, 3, 4))
        f2 = [n.GetIdx() for n in m.GetAtomWithIdx(f).GetNeighbors() if n.GetIdx() not in (r, f1) and n.GetSymbol() != 'C' or False]
        f2 = [n.GetIdx() for n in m.GetAtomWithIdx(f).GetNeighbors() if n.GetIdx() not in (r, f1)]
        r2 = [n.GetIdx() for n in m.GetAtomWithIdx(r).GetNeighbors() if n.GetIdx() not in (f, r1)]
        if key == 'allene':  # アレンは軸の端が C1/C3、中央 C を通して隣の原子へ
            f2 = [n.GetIdx() for n in m.GetAtomWithIdx(f).GetNeighbors() if n.GetSymbol() == 'H']
            r2 = [n.GetIdx() for n in m.GetAtomWithIdx(r).GetNeighbors() if n.GetSymbol() == 'H']
        f2, r2 = f2[0], r2[0]
        # 伸びた四面体で CIP(手前 > 奥)
        d = det_chir(P[f1], P[f2], P[r1], P[r2])
        tor = torsion(P[f1], P[f], P[r], P[r1])
        isR = d < 0
        # aR ≡ M(ねじれ角 < 0) の整合チェック
        assert isR == (tor < 0), (key, d, tor)
        if key == 'allene':
            spec['_ang'] = torsion(P[f1], P[f], P[r], P[r1])
        spec['_code'] = ('Ra' if isR else 'Sa') + f' (tors {tor:.0f})'
        spec['_prio'] = [f1, f2, r1, r2]
        return isR
    if cat == 'helix':
        cents, _ = ring_centroids_path(m, P)
        tors = [torsion(*cents[i:i+4]) for i in range(len(cents) - 3)]
        s = sum(tors)
        spec['_code'] = ('P' if s > 0 else 'M') + f' (tors {s:.0f})'
        return s > 0
    if cat == 'planar':
        A6, A1, A2, pilot = (mapidx(m, k) for k in (5, 6, 7, 10))
        n = np.cross(P[A1] - P[A6], P[A2] - P[A6])
        ccw = np.dot(n, P[pilot] - P[A6]) > 0  # パイロットから見て反時計回り
        spec['_code'] = 'Sp' if ccw else 'Rp'
        return not ccw
    if cat == 'meso':
        codes = [cip_center(m, P, mapidx(m, k)) for k in (9, 8)]
        spec['_code'] = '/'.join(codes)
        assert sorted(codes) == ['R', 'S'], (key, codes)
        return True


def helix_selftest():
    pts = [np.array([math.cos(t), math.sin(t), .3*t]) for t in (0, 1, 2, 3)]
    assert torsion(*pts) > 0  # 右巻きらせんは正のねじれ


def fix_allene(m, conf):
    """RDKit はアレンの直交構造を再現しないので手で組む(C=C=C を x 軸、両端の置換基面を直交)"""
    c2, c4 = mapidx(m, 1), mapidx(m, 2)
    c3 = [n.GetIdx() for n in m.GetAtomWithIdx(c2).GetNeighbors() if n.GetIdx() in [x.GetIdx() for x in m.GetAtomWithIdx(c4).GetNeighbors()]][0]
    pos = {c2: (-1.31, 0, 0), c3: (0, 0, 0), c4: (1.31, 0, 0)}
    def subs(c, sign, plane):
        me = [n.GetIdx() for n in m.GetAtomWithIdx(c).GetNeighbors() if n.GetIdx() != c3]
        me.sort(key=lambda i: m.GetAtomWithIdx(i).GetSymbol() == 'H')  # CH3 を先に
        for k, i in enumerate(me):
            ang = math.radians(60) * (1 if k == 0 else -1)
            d = 1.50 if m.GetAtomWithIdx(i).GetSymbol() == 'C' else 1.09
            v = [sign * d * math.cos(ang), 0, 0]
            v[plane] = d * math.sin(ang)
            pos[i] = (pos[c][0] + v[0], v[1], v[2])
            if m.GetAtomWithIdx(i).GetSymbol() == 'C':
                hs = [n.GetIdx() for n in m.GetAtomWithIdx(i).GetNeighbors() if n.GetSymbol() == 'H']
                for j, h in enumerate(hs):
                    pos[h] = (pos[i][0] + sign * .4, pos[i][1] + .3 * math.cos(j * 2.1), pos[i][2] + .3 * math.sin(j * 2.1))
    subs(c2, -1, 1)
    subs(c4, 1, 2)
    for i, p in pos.items():
        conf.SetAtomPosition(i, p)


def helicene():
    """[6]ヘリセン:コロネンの外周6環を1か所で切り開き、らせん状に持ち上げた形を解析的に組む"""
    a, pitch = 1.40, 3.3
    pts, rings = [], []
    def idx(p):
        for i, q in enumerate(pts):
            if np.linalg.norm(q - p) < .05: return i
        pts.append(p); return len(pts) - 1
    for i in range(6):
        c = a * math.sqrt(3) * np.array([math.cos(math.radians(60*i)), math.sin(math.radians(60*i))])
        ring = []
        for k in range(6):
            v = c + a * np.array([math.cos(math.radians(60*i + 30 + 60*k)), math.sin(math.radians(60*i + 30 + 60*k))])
            ang = math.degrees(math.atan2(v[1], v[0]))
            # 環 i の中での連続角(-30..330)
            while ang < 60*i - 90: ang += 360
            while ang > 60*i + 90: ang -= 360
            ring.append(idx(np.array([v[0], v[1], pitch * (ang + 30) / 360])))
        rings.append(ring)
    rw = Chem.RWMol()
    for _ in pts:
        at = Chem.Atom(6); at.SetIsAromatic(True); rw.AddAtom(at)
    seen = set()
    for r in rings:
        for k in range(6):
            i, j = sorted((r[k], r[(k+1) % 6]))
            if (i, j) not in seen:
                seen.add((i, j)); rw.AddBond(i, j, Chem.BondType.AROMATIC)
                rw.GetBondBetweenAtoms(i, j).SetIsAromatic(True)
    m = rw.GetMol(); Chem.SanitizeMol(m)
    top = max(range(6), key=lambda k: np.mean([pts[i][2] for i in rings[k]]))
    m.GetAtomWithIdx(rings[top][0]).SetAtomMapNum(9)
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(pts): conf.SetAtomPosition(i, p.tolist())
    m.AddConformer(conf)
    assert m.GetNumAtoms() == 26
    return m


def pcp():
    """4-ブロモ[2.2]パラシクロファン:ひずみが大きく ETKDG で埋め込めないので解析的に組む"""
    rw = Chem.RWMol(); P = []
    def add(el, p, arom=False, mp=0):
        at = Chem.Atom(el); at.SetIsAromatic(arom)
        if mp: at.SetAtomMapNum(mp)
        P.append(p); return rw.AddAtom(at)
    rA, rB = [], []
    for zs, ring, deck in ((1, rA, 'A'), (-1, rB, 'B')):
        for k in range(6):
            th = math.radians(60 * k)
            z = zs * (1.39 if k in (0, 3) else 1.55)
            mp = 0
            if deck == 'A': mp = {0: 5, 1: 6, 2: 7, 4: 9}.get(k, 0)
            ring.append(add(6, (1.39*math.cos(th), 1.39*math.sin(th), z), True, mp))
        for k in range(6):
            rw.AddBond(ring[k], ring[(k+1) % 6], Chem.BondType.AROMATIC)
    for xs, ka in ((1, 0), (-1, 3)):
        c1 = add(6, (xs*2.75, 0, .78), mp=10 if xs == 1 else 0)
        c2 = add(6, (xs*2.75, 0, -.78))
        rw.AddBond(rA[ka], c1, Chem.BondType.SINGLE); rw.AddBond(c1, c2, Chem.BondType.SINGLE); rw.AddBond(c2, rB[ka], Chem.BondType.SINGLE)
    th = math.radians(60)
    br = add(35, (3.29*math.cos(th), 3.29*math.sin(th), 1.55))
    rw.AddBond(rA[1], br, Chem.BondType.SINGLE)
    m = rw.GetMol(); Chem.SanitizeMol(m)
    m = Chem.AddHs(m)  # 水素は描かないが原子ラベルの H 数計算用(座標は使わない)
    conf = Chem.Conformer(m.GetNumAtoms())
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, P[i] if i < len(P) else P[m.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx()])
    m.AddConformer(conf)
    return m


def build(key, spec):
    if key == 'helicene':
        return build_mol(key, spec, helicene())
    if key == 'pcp':
        return build_mol(key, spec, pcp())
    m0 = Chem.MolFromSmiles(spec['smi'])
    m = Chem.AddHs(m0)
    ps = AllChem.ETKDGv3(); ps.randomSeed = 7; ps.useRandomCoords = True
    assert AllChem.EmbedMolecule(m, ps) == 0, key
    AllChem.MMFFOptimizeMolecule(m, maxIters=5000)
    conf = m.GetConformer()
    if key == 'allene':
        fix_allene(m, conf)
    return build_mol(key, spec, m)


def build_mol(key, spec, m):
    conf = m.GetConformer()
    if spec['cat'] == 'meso':
        c2, c3 = mapidx(m, 9), mapidx(m, 8)
        s1 = [n.GetIdx() for n in m.GetAtomWithIdx(c2).GetNeighbors() if n.GetSymbol() == spec['syn'][0]][0]
        s2 = [n.GetIdx() for n in m.GetAtomWithIdx(c3).GetNeighbors() if n.GetSymbol() == spec['syn'][1]][0]
        rdMolTransforms.SetDihedralDeg(conf, s1, c2, c3, s2, 0.0)
    P = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
    if spec['cat'] == 'center':
        spec['_center'] = [a.GetIdx() for a in m.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED][0]
    # 右の箱側に揃える(鏡映)
    if not descriptor(key, spec, m, P):
        P[:, 2] *= -1
        assert descriptor(key, spec, m, P), key
    code_before = spec['_code']

    # ---- 向き ----
    cat = spec['cat']
    cen = P.mean(axis=0)
    if cat == 'axial':
        f, r, f1 = mapidx(m, 1), mapidx(m, 2), mapidx(m, 3)
        ax = P[r] - P[f]; ax /= np.linalg.norm(ax)
        side = P[f1] - P[f]; side -= np.dot(side, ax) * ax
        # 軸を縦(y 上向き = 手前原子が上)、手前環を少し斜めに
        R0 = frame(side, -ax)
        R = rot([0, 1, 0], spec.get('phi', 45)) @ R0
        R = rot([1, 0, 0], spec.get('tilt', 22)) @ R
        if key == 'allene':
            R = rot([0, 0, 1], 90) @ R
    elif cat == 'planar':
        A6, A1 = mapidx(m, 5), mapidx(m, 6)
        ringA = [a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic() and a.IsInRingSize(6)]
        ri = m.GetRingInfo()
        rA = [list(x) for x in ri.AtomRings() if A1 in x][0]
        rB = [list(x) for x in ri.AtomRings() if len(x) == 6 and not set(x) & set(rA) and all(m.GetAtomWithIdx(i).GetIsAromatic() for i in x)][0]
        up = P[rA].mean(axis=0) - P[rB].mean(axis=0)
        bridge = [n for n in rA if any(nb.GetSymbol() == 'C' and not nb.GetIsAromatic() for nb in m.GetAtomWithIdx(n).GetNeighbors())]
        xdir = P[bridge[1]] - P[bridge[0]]
        R = frame(xdir, up)
        R = rot([0, 1, 0], 180) @ R  # 環の面内で半回転して Br を手前の縁へ
        R = rot([1, 0, 0], spec.get('tilt', 48)) @ R
        R = rot([0, 1, 0], spec.get('yaw', 0)) @ R  # 少し斜めから見て Br が外に出るように
    elif cat == 'meso':
        c2, c3 = mapidx(m, 9), mapidx(m, 8)
        xdir = P[c3] - P[c2]
        s1 = [n.GetIdx() for n in m.GetAtomWithIdx(c2).GetNeighbors() if n.GetSymbol() == spec['syn'][0]][0]
        mid = (P[c2] + P[c3]) / 2
        R = frame(xdir, P[s1] - P[c2])
        R = rot([1, 0, 0], -8) @ R
        cen = mid
    else:
        heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() != 'H']
        X = P[heavy] - P[heavy].mean(axis=0)
        w, v = np.linalg.eigh(X.T @ X)
        v = v[:, ::-1]
        if np.linalg.det(v) < 0:
            v[:, 2] *= -1
        R = v.T
        if cat == 'center':
            h = [n.GetIdx() for n in m.GetAtomWithIdx(spec['_center']).GetNeighbors() if n.GetSymbol() == 'H']
            if h and (R @ (P[h[0]] - cen))[2] > 0:
                R = rot([0, 1, 0], 180) @ R  # H を奥へ
            if h:  # 顔に隠れないよう、H を「奥・下向きのしっぽ」として少し見せる
                c0 = spec['_center']
                best = None
                for deg in sorted(range(-90, 91, 5), key=abs):
                    Rt = rot([1, 0, 0], deg) @ R
                    v = Rt @ (P[h[0]] - P[c0]); n = np.linalg.norm(v)
                    if v[2] < 0 and v[1] < 0 and math.hypot(v[0], v[1]) / n >= .62:
                        best = Rt; break
                if best is not None: R = best
        for axn, deg in spec.get('rot', []):
            R = rot(axn, deg) @ R
    assert abs(np.linalg.det(R) - 1) < 1e-6
    Q = (R @ (P - cen).T).T
    assert descriptor(key, spec, m, Q), key
    assert spec['_code'].split()[0] == code_before.split()[0]

    # ---- 出力する原子 ----
    keepH = set()
    if cat in ('center', 'meso'):
        for c in ([spec['_center']] if cat == 'center' else [mapidx(m, 9), mapidx(m, 8)]):
            keepH |= {n.GetIdx() for n in m.GetAtomWithIdx(c).GetNeighbors() if n.GetSymbol() == 'H'}
    if key == 'allene':
        keepH |= {n.GetIdx() for k in (1, 2) for n in m.GetAtomWithIdx(mapidx(m, k)).GetNeighbors() if n.GetSymbol() == 'H'}
    keep = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() != 'H' or a.GetIdx() in keepH]
    if key == 'diphenic':  # ニトロ基は NO₂ ラベルにまとめる
        keep = [i for i in keep if not (m.GetAtomWithIdx(i).GetSymbol() == 'O' and any(n.GetSymbol() == 'N' for n in m.GetAtomWithIdx(i).GetNeighbors()))]
    newi = {o: i for i, o in enumerate(keep)}
    Chem.Kekulize(m, clearAromaticFlags=False)
    atoms = []
    for o in keep:
        a = m.GetAtomWithIdx(o)
        sym = a.GetSymbol()
        lab = ''
        if sym not in ('C',):
            nh = sum(1 for n in a.GetNeighbors() if n.GetSymbol() == 'H' and n.GetIdx() not in keepH)
            lab = sym + ('H' if nh else '') + (str(nh).translate(SUB) if nh > 1 else '')
            if key != 'diphenic':
                if a.GetFormalCharge() > 0: lab += '⁺'
                if a.GetFormalCharge() < 0: lab += '⁻'
            lab = LABEL_OVERRIDE.get(key, {}).get(sym, lab)
        x, y, z = Q[o] * PX
        atoms.append([sym, lab, round(x, 1), round(-y, 1), round(z, 1)])  # 画面系: y 下向き
    bonds = []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in newi and j in newi:
            bt = b.GetBondType()
            o = 2 if bt == Chem.BondType.DOUBLE else 3 if bt == Chem.BondType.TRIPLE else 1
            bonds.append([newi[i], newi[j], o])
    # 塗る・顔を置くのは普通の環(6員環以下)だけ。橋を回る大環状の「環」は除く
    rings = [[newi[i] for i in r] for r in m.GetRingInfo().AtomRings() if len(r) <= 6 and all(i in newi for i in r)]
    faces = []
    for mp in (9, 8):
        ai = mapidx(m, mp)
        if ai is None: continue
        rr_ = [k for k, r in enumerate(rings) if newi[ai] in r]
        if rr_ and cat != 'meso':
            faces.append({'ring': min(rr_, key=lambda k: len(rings[k]))})
        else:
            faces.append({'atom': newi[ai]})
    if key == 'helicene':  # 顔は手前側の末端の環に
        term = [k for k, r in enumerate(rings) if sum(1 for r2 in rings if r2 is not r and len(set(r) & set(r2)) == 2) == 1]
        faces = [{'ring': max(term, key=lambda k: np.mean([Q[keep[i]][2] for i in rings[k]]))}]
    out = {'a': atoms, 'b': bonds, 'r': rings, 'f': faces}
    # やさしいモード用の優先順位
    if cat == 'center':
        c = spec['_center']
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
        nb = [n for n in m.GetAtomWithIdx(c).GetNeighbors()]
        nb.sort(key=lambda n: -int(n.GetProp('_CIPRank')))
        out['st'] = [newi[c], [newi[n.GetIdx()] for n in nb if n.GetIdx() in newi]]
    if cat == 'axial':
        out['st'] = [None, [newi[i] for i in spec['_prio']]]
    if cat == 'meso':
        out['st'] = [newi[mapidx(m, 9)], []]
    ext = max(math.hypot(a[2], a[3]) for a in atoms)
    out['e'] = round(ext, 1)
    print(f"{key:15s} {spec['_code']:18s} atoms={len(atoms):3d} ext={ext:5.1f}px", file=sys.stderr)
    return out


helix_selftest()
res = {k: build(k, v) for k, v in MOLS.items()}
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else 'mols.json', 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
