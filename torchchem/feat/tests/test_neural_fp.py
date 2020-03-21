import unittest
from rdkit import Chem
from rdkit.Chem import AllChem
import operator
from torchchem.feat.neural_fp import fillBondType_dic
from torchchem.feat.neural_fp import fillAtomType_dic
from torchchem.feat.neural_fp import molToGraph


class TestNeuralFp(unittest.TestCase):

  def test_mol_to_graph(self):
    """Test conversion of molecule to graph."""
    mol = Chem.MolFromSmiles("CC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, maxAttempts=5000)
    AllChem.UFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)

    bondtype_list_order = []
    atomtype_list_order = []
    bondtype_dic = {}
    atomtype_dic = {}
    bondtype_dic = fillBondType_dic(mol, bondtype_dic)
    atomtype_dic = fillAtomType_dic(mol, atomtype_dic)

    sorted_bondtype_dic = sorted(
        bondtype_dic.items(), key=operator.itemgetter(1))
    sorted_bondtype_dic.reverse()
    bondtype_list_order = [ele[0] for ele in sorted_bondtype_dic]
    sorted_atom_types_dic = sorted(
        atomtype_dic.items(), key=operator.itemgetter(1))
    sorted_atom_types_dic.reverse()
    atomtype_list_order = [ele[0] for ele in sorted_atom_types_dic]
    graph = molToGraph(
        mol,
        bondtype_list_order,
        atomtype_list_order,
        molecular_attributes=False,
        formal_charge_one_hot=False)
