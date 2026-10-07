from rdkit import Chem
from rdkit.Chem import rdchem
import traceback
from tqdm import tqdm

def AdjustAromaticNs(m, nitrogenPattern="[n&D2&H0;r5,r6]"):
    """
    default nitrogen pattern matches Ns in 5 rings and 6 rings in order to be able
    to fix: O=c1ccncc1
    """
    Chem.GetSymmSSSR(m)
    m.UpdatePropertyCache(False)

    # break non-ring bonds linking rings:
    em = Chem.EditableMol(m)
    linkers = m.GetSubstructMatches(Chem.MolFromSmarts("[r]!@[r]"))
    plsFix = set()
    for a, b in linkers:
        em.RemoveBond(a, b)
        plsFix.add(a)
        plsFix.add(b)
    nm = em.GetMol()
    for at in plsFix:
        at = nm.GetAtomWithIdx(at)
        if at.GetIsAromatic() and at.GetAtomicNum() == 7:
            at.SetNumExplicitHs(1)
            at.SetNoImplicit(True)

    # build molecules from the fragments:
    fragLists = Chem.GetMolFrags(nm)
    frags = [_FragIndicesToMol(nm, x) for x in fragLists]

    # loop through the fragments in turn and try to aromatize them:
    ok = True
    for i, frag in enumerate(frags):
        cp = Chem.Mol(frag)
        try:
            Chem.SanitizeMol(cp)
        except ValueError:
            matches = [
                x[0]
                for x in frag.GetSubstructMatches(Chem.MolFromSmarts(nitrogenPattern))
            ]
            lres, indices = _recursivelyModifyNs(frag, matches)
            if not lres:
                # print 'frag %d failed (%s)'%(i,str(fragLists[i]))
                ok = False
                break
            else:
                revMap = {}
                for k, v in frag._idxMap.iteritems():
                    revMap[v] = k
                for idx in indices:
                    oatom = m.GetAtomWithIdx(revMap[idx])
                    oatom.SetNoImplicit(True)
                    oatom.SetNumExplicitHs(1)
    return m


class MoleculesHandler:
    def __init__(self, num_atoms, ontology):
        self.ontology = ontology
        self.num_atoms = num_atoms
        self.atom_types = {
            p.name for p in self.ontology.predicates.values() if len(p.domains) == 1
        }
        self.bond_types = {
            p.name
            for p in self.ontology.predicates.values()
            if len(p.domains) == 2 and p.name != "skipBond"
        }

        self.fol_to_RDKIT = {
            p.name: p.name.upper() for p in self.ontology.predicates.values()
        }
        self.fol_to_RDKIT["cl"] = "Cl"
        self.RDKIT_to_fol = {v: k for k, v in self.fol_to_RDKIT.items()}

    def smile2Fol(self, smile):
        mol = Chem.MolFromSmiles(smile)

        to_write = []

        if "aromatic" not in self.bond_types:
            Chem.Kekulize(mol, clearAromaticFlags=True)

        for a in mol.GetAtoms():
            sym = self.RDKIT_to_fol[str(a.GetSymbol())]

            aid = a.GetIdx()
            to_write.append("%s(%d)" % (sym, aid))

            if len(to_write) == 0:
                continue

            for b in mol.GetBonds():
                sid = b.GetBeginAtom().GetIdx()
                eid = b.GetEndAtom().GetIdx()
                type = self.RDKIT_to_fol[str(b.GetBondType())]
                to_write.append("%s(%d,%d)" % (type, sid, eid))
                to_write.append("%s(%d,%d)" % (type, eid, sid))

            print(to_write)

    def fromLin2Mol(self, Y):
        mols = []
        for linear in tqdm(Y, desc="fromLin2Mol"):
            d = self.ontology.linear_to_fol_dictionary(linear)
            try:
                mol = self.fromFol2Mol(d)
            except Exception as e:
                print("Exception occurred:", e)
                traceback.print_exc()
                mol = None
            if mol is not None:
                mols.append(mol)
        return mols

    def fromFol2Mol(self, d, sanitize=True):
        mol = Chem.RWMol()
        node_to_idx = {}

        for i in range(self.num_atoms):

            S = None
            for s in self.atom_types:
                if d[s][i] == 1:
                    S = s
                    break
            a = Chem.Atom(self.fol_to_RDKIT[S])

            idx = mol.AddAtom(a)
            node_to_idx[i] = idx

        for i in range(self.num_atoms):
            for j in range(i, self.num_atoms):
                if i not in node_to_idx or j not in node_to_idx:
                    continue
                for b in self.bond_types:
                    if d[b][i, j] == 1:
                        ifirst = node_to_idx[i]
                        isecond = node_to_idx[j]
                        bond_type = rdchem.BondType.names[self.fol_to_RDKIT[b]]
                        mol.AddBond(ifirst, isecond, bond_type)
                        break

        mol = mol.GetMol()
        if sanitize:
            mol = AdjustAromaticNs(mol)
            mol.UpdatePropertyCache(False)
            for a in mol.GetAtoms():
                a.UpdatePropertyCache()
            Chem.Kekulize(mol)
            Chem.SanitizeMol(mol)
        return mol


def _recursivelyModifyNs(mol, matches, indices=None):
    if indices is None:
        indices = []
    res = None
    while len(matches) and res is None:
        tIndices = indices[:]
        nextIdx = matches.pop(0)
        tIndices.append(nextIdx)
        nm = Chem.Mol(mol)
        nm.GetAtomWithIdx(nextIdx).SetNoImplicit(True)
        nm.GetAtomWithIdx(nextIdx).SetNumExplicitHs(1)
        cp = Chem.Mol(nm)
        try:
            Chem.SanitizeMol(cp)
        except ValueError:
            res, indices = _recursivelyModifyNs(nm, matches, indices=tIndices)
        else:
            indices = tIndices
            res = cp
    return res, indices


def _FragIndicesToMol(oMol, indices):
    em = Chem.EditableMol(Chem.Mol())

    newIndices = {}
    for i, idx in enumerate(indices):
        em.AddAtom(oMol.GetAtomWithIdx(idx))
        newIndices[idx] = i

    for i, idx in enumerate(indices):
        at = oMol.GetAtomWithIdx(idx)
        for bond in at.GetBonds():
            if bond.GetBeginAtomIdx() == idx:
                oidx = bond.GetEndAtomIdx()
            else:
                oidx = bond.GetBeginAtomIdx()
            # make sure every bond only gets added once:
            if oidx < idx:
                continue
            em.AddBond(newIndices[idx], newIndices[oidx], bond.GetBondType())
    res = em.GetMol()
    res.ClearComputedProps()
    Chem.GetSymmSSSR(res)
    res.UpdatePropertyCache(False)
    res._idxMap = newIndices
    return res
