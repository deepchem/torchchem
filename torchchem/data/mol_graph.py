"""
Molecule graph featurizer
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.Descriptors as Descriptors
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.EState as EState
import rdkit.Chem.rdPartialCharges as rdPartialCharges
import rdkit.Chem.rdChemReactions as rdRxns
import copy

import torch as t
import torch_geometric as tg

symbol_list = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'Other']

hybridization_list = [
      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, 
      Chem.rdchem.HybridizationType.SP3D2, 'Other'
      ]

bond_list = [
    Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
    'Other'
    ]

bond_stereo_list = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE
    ]

def one_of_k_encoding(x, allowable_set):
  """Maps inputs to one hot vectors."""
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))

# def safe_index(l, e):
#   """Gets the index of e in l, providing an index of len(l) if not found"""
#   try:
#     return l.index(e)
#   except:
#     return len(l)
  

def get_atom_features(atom,
                      explicit_H=False,
                      use_chirality=False):
  """
  Generate feature vector for each atom:
    [atom type,
     degree (number of neighbors),
     valence,
     hybridization state,
     formal charge,
     radical electrons,
     if in ring,
     if in aromatic ring,
     (number of Hs),
     (chirality)]

  Parameters
  ----------
  atom : rdkit.Chem.rdchem.Atom
    rdkit atom class
  explicit_H : bool, optional
    If H should be explicitly included as nodes. The default is False.
  use_chirality : bool, optional
    If to add chirality features. The default is False.

  Returns
  -------
  atom_feat : list
    feature vector.

  """
  # Atom type
  atom_symbol_feat = one_of_k_encoding_unk(atom.GetSymbol(), symbol_list)
  # Number of neighbors
  atom_degree_feat = one_of_k_encoding(atom.GetDegree(), list(range(10)))
  # Valence
  atom_valence_feat = one_of_k_encoding(atom.GetImplicitValence(), 
                                        list(range(-1, 7)))
  # Hybridization
  atom_hybrid_feat = one_of_k_encoding_unk(atom.GetHybridization(), 
                                           hybridization_list)
  atom_feat = atom_symbol_feat + atom_degree_feat + atom_valence_feat + atom_hybrid_feat
  # Charge, radicals, ring info
  atom_feat += [atom.GetFormalCharge(), 
                atom.GetNumRadicalElectrons(), 
                atom.IsInRing(),
                atom.GetIsAromatic()]
  
  if not explicit_H:
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    atom_feat += one_of_k_encoding_unk(atom.GetTotalNumHs(), 
                                       [0, 1, 2, 3, 4, 'Other'])
  if use_chirality:
      try:
        atom_feat += one_of_k_encoding(atom.GetProp('_CIPCode'), ['R', 'S'])
      except:
        atom_feat += [False] * 2
  return atom_feat

def get_atom_features_length(explicit_H=False, use_chirality=False):
  """
  Get length of atom features
  """
  length = len(symbol_list) + 10 + 8 + len(hybridization_list) + 4
  if not explicit_H:
    length += 6
  if use_chirality:
    length += 2
  return length

def get_bond_features(bond, use_chirality=False):
  """
  Generate feature vector for each bond:
    [bond type,
     if conjugated,
     if in ring,
     (chirality)]

  Parameters
  ----------
  bond : rdkit.Chem.rdchem.Bond
    rdkit bond class
  use_chirality : bool, optional
    If to add chirality features. The default is False.

  Returns
  -------
  bond_feat : list
    feature vector.

  """
  bond_feat = one_of_k_encoding_unk(bond.GetBondType(), bond_list)
  bond_feat += [bond.GetIsConjugated(), bond.IsInRing()]
  if use_chirality:
      try:
        bond_feat += one_of_k_encoding_unk(bond.GetStereo(), bond_stereo_list)
      except:
        bond_feat += [False] * 4
  return bond_feat

def get_bond_features_length(use_chirality=False):
  """
  Get length of bond features
  """
  length = len(bond_list) + 2
  if use_chirality:
    length += len(bond_stereo_list)
  return length

def get_adjacency_list(rdmol):
  """
  Generate adjacency list for molecule
  """
  canon_adj_list = [[] for i in range(rdmol.GetNumAtoms())]
  for bond in rdmol.GetBonds():
    begin_atom = bond.GetBeginAtomIdx()
    end_atom = bond.GetEndAtomIdx()
    canon_adj_list[begin_atom].append(end_atom)
    canon_adj_list[end_atom].append(begin_atom)
  return canon_adj_list

def get_pair_features(rdmol, graph_distance=True, use_chirality=False):
  """
  Generate feature vector for each atom-atom pair:
    [bond features,
     if in the same ring,
     graph distance/catesian distance]

  Parameters
  ----------
  rdmol : rdkit.Chem.rdchem.Mol
    rdkit molecule class
  graph_distance : bool, optional
    If to use graph distance or catesian distance. The default is True.
  use_chirality : bool, optional
    If to add chirality features. The default is False.

  Returns
  -------
  pair_feats : np.array
    feature matrix.
    
  """
  num_atoms = rdmol.GetNumAtoms()
  if graph_distance:
    max_distance = 7
    canon_adj_list = get_adjacency_list(rdmol)
  else:
    max_distance = 1

  # Get length of bond features
  bond_feat_len = get_bond_features_length(use_chirality=use_chirality)

  bond_feats = np.zeros((num_atoms, num_atoms, bond_feat_len))
  ring_feats = np.zeros((num_atoms, num_atoms, 1))
  distance_feats = np.zeros((num_atoms, num_atoms, max_distance))

  rings = rdmol.GetRingInfo().AtomRings()
  for a1 in range(num_atoms):
    for a2 in range(num_atoms):
      bond = rdmol.GetBondBetweenAtoms(a1, a2)
      if not bond is None:
        bond_feats[a1, a2] = np.array(get_bond_features(
            bond, 
            use_chirality=use_chirality))
    for ring in rings:
      if a1 in ring:
        ring_feats[a1, ring, 0] = 1
        ring_feats[a1, a1, 0] = 0.
    # Graph distance between two atoms
    if graph_distance:
      distance = find_distance(
          a1, num_atoms, canon_adj_list, max_distance=max_distance)
      distance_feats[a1, :, :] = distance
  # Euclidean distance between atoms
  if not graph_distance:
    coords = np.zeros((num_atoms, 3))
    for a1 in range(num_atoms):
      pos = rdmol.GetConformer(0).GetAtomPosition(a1)
      coords[a1, :] = [pos.x, pos.y, pos.z]
    distance_feats[:, :, 0] = np.sqrt(np.sum(np.square(
      np.stack([coords] * num_atoms, axis=1) - \
      np.stack([coords] * num_atoms, axis=0)), axis=2))

  pair_feats = np.concatenate([bond_feats, ring_feats, distance_feats], 2)
  return pair_feats

def find_distance(a1, num_atoms, canon_adj_list, max_distance=7):
  """
  Calculate graph distance between atom a1 with the remaining atoms using BFS
  """
  distance = np.zeros((num_atoms, max_distance))
  radial = 0
  # atoms `radial` bonds away from `a1`
  adj_list = set(canon_adj_list[a1])
  # atoms less than `radial` bonds away
  all_list = set([a1])
  while radial < max_distance:
    distance[list(adj_list), radial] = 1
    all_list.update(adj_list)
    # find atoms `radial`+1 bonds away
    next_adj = set()
    for adj in adj_list:
      next_adj.update(canon_adj_list[adj])
    adj_list = next_adj - all_list
    radial = radial + 1
  return distance

def get_molecular_attributes(rdmol):
  """
  Molecular attributes calculated as:
    [Crippen contribution to logp,
     Crippen contribution to mr,
     TPSA contribution,
     Labute ASA contribution,
     EState Index,
     Gasteiger partial charge,
     Gasteiger hydrogen partial charge]

  Parameters
  ----------
  rdmol : rdkit.Chem.rdchem.Mol
    rdkit molecule class

  Returns
  -------
  attributes : list
    feature vector

  """
  attributes = [[] for _ in rdmol.GetAtoms()]
  
  for (i, x) in enumerate(rdMolDescriptors._CalcCrippenContribs(rdmol)):
    attributes[i].append(x[0])
    attributes[i].append(x[1])
  for (i, x) in enumerate(rdMolDescriptors._CalcTPSAContribs(rdmol)):
    attributes[i].append(x)
  for (i, x) in enumerate(rdMolDescriptors._CalcLabuteASAContribs(rdmol)[0]):
    attributes[i].append(x)
  for (i, x) in enumerate(EState.EStateIndices(rdmol)):
    attributes[i].append(x)
    
  rdPartialCharges.ComputeGasteigerCharges(rdmol)
  for (i, a) in enumerate(rdmol.GetAtoms()):
    val = float(a.GetProp('_GasteigerCharge'))
    if val == val and val < np.inf:
      attributes[i].append(val)
    else:
      attributes[i].append(0.0)
  for (i, a) in enumerate(rdmol.GetAtoms()):
    val = float(a.GetProp('_GasteigerHCharge'))
    if val == val and val < np.inf:
      attributes[i].append(val)
    else:
      attributes[i].append(0.0)

  return attributes


def mol_to_graph(rdmol,
                 explicit_H=False,
                 use_chirality=False,
                 use_molecular_attributes=False,
                 all_pair_features=False,
                 graph_distance=True):
  """
  Converts an RDKit molecule to a graph

  Parameters
  ----------
  rdmol : rdkit.Chem.rdchem.Mol
    rdkit molecule class
  explicit_H : bool, optional
    If H should be explicitly included as nodes. The default is False.
  use_chirality : bool, optional
    If to add chirality features. The default is False.
  use_molecular_attributes : bool, optional
    If to add molecular properties calculated by RDkit. The default is False.
  all_pair_features : bool, optional
    If True, molecules are regarded as densely-connected graph.
    If False, only bonds are regarded as edges. The default is False.
  graph_distance : bool, optional
    If to use graph distance or catesian distance. The default is True.
    
  Returns
  -------
  mol_graph : tg.data.Data
    Molecule graph

  """

  # Calculate node (atom) features
  num_atoms = rdmol.GetNumAtoms()
  node_features = []
  for i in range(num_atoms):
    atom = rdmol.GetAtomWithIdx(i)
    node_features.append(get_atom_features(
        atom,
        explicit_H=explicit_H,
        use_chirality=use_chirality))
  node_features = np.array(node_features).astype(float)
  assert node_features.shape[0] == num_atoms
  assert node_features.shape[1] == get_atom_features_length(
      explicit_H=explicit_H,
      use_chirality=use_chirality)

  if use_molecular_attributes:
    extra_attributes = np.array(get_molecular_attributes(rdmol)).astype(float)
    assert extra_attributes.shape[0] == num_atoms
    node_features = np.concatenate([node_features, extra_attributes], 1)

  # Calculate edge (bond/pair) features
  edge_features = []
  edge_index = []
  if not all_pair_features:
    # Only use bonds as edges
    for bond in rdmol.GetBonds():
      begin_atom = bond.GetBeginAtomIdx()
      end_atom = bond.GetEndAtomIdx()
      bond_features = get_bond_features(
          bond, 
          use_chirality=use_chirality)
      # Add two edges to force undirected graph
      edge_features.append(bond_features)
      edge_index.append((begin_atom, end_atom))
      edge_features.append(bond_features)
      edge_index.append((end_atom, begin_atom))
    bond_feat_len = get_bond_features_length(use_chirality=use_chirality)
    edge_features = np.array(edge_features).reshape((-1, bond_feat_len)).astype(float)
    edge_index = np.array(edge_index).reshape((-1, 2)).astype(int)
    edge_index = edge_index.transpose((1, 0))
  else:
    # Use dense connections
    all_pair_feats = get_pair_features(rdmol, 
                                       graph_distance=graph_distance, 
                                       use_chirality=use_chirality)
    for a1 in range(num_atoms):
      for a2 in range(num_atoms):
        if a1 == a2:
          continue
        edge_features.append(all_pair_feats[a1, a2])
        edge_index.append((a1, a2))
    edge_features = np.array(edge_features).astype(float)
    edge_index = np.stack(edge_index, 1).astype(int)

  mol_graph = tg.data.Data(x=t.from_numpy(node_features).float(),
                           edge_index=t.from_numpy(edge_index).long(),
                           edge_attr=t.from_numpy(edge_features).float())
  assert mol_graph.is_undirected()
  return mol_graph