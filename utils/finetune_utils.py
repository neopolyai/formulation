import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
from chemutils import get_mol, get_clique_mol

# Declare possible molecular properties
features = {
    'atomic_num': [0,1,5,6,7,8,9,11,14,15,16,17,19,20,26,27,28,30,35,53],
    'degree' : [0,1,2,3,4,5,6],
    'formal_charge' : [-3,-2,-1,0,1,2,3],
    'chirality' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER],
    'bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC],
    'bond_inring': [
        None, 
        False, 
        True],
    'bond_isconjugated': [
        None,
        False,
        True],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOZ]
}
MAX_ATOM_TYPE = len(features['atomic_num'])



def get_gasteiger_partial_charges(mol, n_iter=12):
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter, throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms()]
    return partial_charges


    
def mol_to_graph_data_obj_simple(mol, smiles):
    # Get atom features
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [
            features['atomic_num'].index(atom.GetAtomicNum())] + [
                features['degree'].index(atom.GetDegree())] + [
                    features['chirality'].index(atom.GetChiralTag())] + [
                        features['formal_charge'].index(atom.GetFormalCharge())]
        atom_features_list.append(atom_feature)

    # Get edge indices and edge features
    edges_list = []
    edge_features_list = []
    for bond in mol.GetBonds():
        # Append start and end atoms of the bond
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges_list.append((i, j))
        edges_list.append((j, i))
        # Append features/attributes of the bond
        edge_feature = [
            features['bonds'].index(bond.GetBondType())] + [
                features['bond_inring'].index(bond.IsInRing())] + [
                    features['stereo'].index(bond.GetStereo())] + [
                        features['bond_isconjugated'].index(bond.GetIsConjugated())]
        edge_features_list.append(edge_feature)
        edge_features_list.append(edge_feature)
    
    # Tesorize features and indices
    x_nosuper = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    edge_index_nosuper = torch.tensor(np.array(edges_list).T, dtype=torch.long)
    edge_attr_nosuper = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    # Get num_atoms and num_motifs
    num_atoms = x_nosuper.size(0)    
    cliques = motif_decomp(mol, smiles)
    num_motif = len(cliques)

    if num_motif > 0:
        # Update self.x 
        super_x = torch.tensor([[MAX_ATOM_TYPE, 0, 0, 3]]).to(x_nosuper.device)
        motif_x = torch.tensor([[MAX_ATOM_TYPE+1, 0, 0, 3]]).repeat_interleave(num_motif, dim=0).to(x_nosuper.device)
        x = torch.cat((x_nosuper, motif_x, super_x), dim=0)

        # Create super_edge_index and super_edge_attr (self-loop)
        super_edge_index = [[num_atoms+i, num_atoms+num_motif] for i in range(num_motif)]
        super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)
        super_edge_attr = torch.zeros(num_motif, 4)
        super_edge_attr[:,0] = 5 
        super_edge_attr = super_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)

        # Update self.edge_index with motif edge indices
        motif_edge_index = []
        for k, motif in enumerate(cliques):
            motif_edge_index = motif_edge_index + [[i, num_atoms+k] for i in motif]
        motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)
        edge_index = torch.cat((edge_index_nosuper, motif_edge_index, super_edge_index), dim=1)

        # Update self.edge_index with motif edge attributes
        motif_edge_attr = torch.zeros(motif_edge_index.size()[1], 4)
        motif_edge_attr[:,0] = 6 
        motif_edge_attr = motif_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)
        edge_attr = torch.cat((edge_attr_nosuper, motif_edge_attr, super_edge_attr), dim = 0)

    else:
        # Update x with self-loop
        super_x = torch.tensor([[MAX_ATOM_TYPE, 0, 0, 3]]).to(x_nosuper.device)
        x = torch.cat((x_nosuper, super_x), dim=0)
        # Add self-loop edge index
        super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
        super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)
        edge_index = torch.cat((edge_index_nosuper, super_edge_index), dim=1)
        # Add self-loop edge attribute
        super_edge_attr = torch.zeros(num_atoms, 4)
        super_edge_attr[:,0] = 5 
        super_edge_attr = super_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)
        edge_attr = torch.cat((edge_attr_nosuper, super_edge_attr), dim = 0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data



def motif_decomp(mol, smiles):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    cliques = []  
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])  

    res = list(BRICS.FindBRICSBonds(mol))  
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]]) 

    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0: 
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms> len(c) > 0]

    num_cli = len(cliques)
    ssr_mol = Chem.GetSymmSSSR(mol)
    for i in range(num_cli):
        c = cliques[i]
        cmol = get_clique_mol(mol, c)
        ssr = Chem.GetSymmSSSR(cmol)
        if len(ssr)>1: 
            for ring in ssr_mol:
                if len(set(list(ring)) & set(c)) == len(list(ring)):
                    cliques.append(list(ring))
            cliques[i]=[]
    
    cliques = [c for c in cliques if n_atoms> len(c) > 0]
    return cliques



def _load_dataset(input_path):
    # Load in dataset in CSV format
    input_df = pd.read_csv(input_path, sep=',')
    # api_mol, cosolvent_mol, surfactant_mol, oil1_mol, oil2_mol = [], [], [], [], []
    api_mol, cosolvent_mol, surfactant_mol, oil1_mol = [], [], [], []

    # Convert SMILES to mol objects
    api_smiles = input_df['API_id']
    for s in api_smiles:
        try:
            api_mol.append(AllChem.MolFromSmiles(s))
        except:
            api_mol.append(None)

    cosolvent_smiles = input_df['cosolvent_id']
    for s in cosolvent_smiles:
        try:
            cosolvent_mol.append(AllChem.MolFromSmiles(s))
        except:
            cosolvent_mol.append(None)

    surfactant_smiles = input_df['surfactant_id']
    for s in surfactant_smiles:
        try:
            surfactant_mol.append(AllChem.MolFromSmiles(s))
        except:
            surfactant_mol.append(None)

    oil1_smiles = input_df['oil_id1']
    for s in oil1_smiles:
        try:
            oil1_mol.append(AllChem.MolFromSmiles(s))
        except:
            oil1_mol.append(None)

    # oil2_smiles = input_df['oil_id2']
    # for s in oil2_smiles:
    #     try:
    #         oil2_mol.append(AllChem.MolFromSmiles(s))
    #     except:
    #         oil2_mol.append(None)

    # Compile data together
    # smiles_list = [api_smiles, cosolvent_smiles, surfactant_smiles, oil1_smiles, oil2_smiles]
    # mol_list = [api_mol, cosolvent_mol, surfactant_mol, oil1_mol, oil2_mol]
    smiles_list = [api_smiles, cosolvent_smiles, surfactant_smiles, oil1_smiles]
    mol_list = [api_mol, cosolvent_mol, surfactant_mol, oil1_mol]
    labels = input_df[['size', 'PDI']].fillna(0)   
    properties = input_df[['API_logp', 'API_sol', 'API_polar', 'o_LC', 'o_sat', 's_HLB']].fillna(0)  
    # ratios = input_df[['API_prop', 'cosolvent_prop', 'surfactant_prop', 'oil_prop1', 'oil_prop2']]
    ratios = input_df[['API_prop', 'cosolvent_prop', 'surfactant_prop', 'oil_prop1']].fillna(0)    

    return smiles_list, mol_list, labels.values, ratios.values, properties.values



class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        super(MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.cosolvent_data, self.cosolvent_slices = torch.load('data/training/processed/geometric_data_processed_cosolvent.pt')
        self.surfactant_data, self.surfactant_slices = torch.load('data/training/processed/geometric_data_processed_surfactant.pt')
        self.oil1_data, self.oil1_slices = torch.load('data/training/processed/geometric_data_processed_oil.pt')
        # self.oil2_data, self.oil2_slices = torch.load('data/training/processed/geometric_data_processed_4.pt')

    def get(self, idx):
        # api_data, cosolvent_data, surfactant_data, oil1_data, oil2_data = Data(), Data(), Data(), Data(), Data()
        api_data, cosolvent_data, surfactant_data, oil1_data = Data(), Data(), Data(), Data()

        # Create API data
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[api_data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            api_data[key] = item[s]

        # Create cosolvent data
        for key in self.cosolvent_data.keys():
            cosolvent_item, cosolvent_slices = self.cosolvent_data[key], self.cosolvent_slices[key]
            s = list(repeat(slice(None), cosolvent_item.dim()))
            s[cosolvent_data.__cat_dim__(key, cosolvent_item)] = slice(cosolvent_slices[idx], cosolvent_slices[idx + 1])
            cosolvent_data[key] = cosolvent_item[s]

        # Create surfactant data
        for key in self.surfactant_data.keys():
            surfactant_item, surfactant_slices = self.surfactant_data[key], self.surfactant_slices[key]
            s = list(repeat(slice(None), surfactant_item.dim()))
            s[surfactant_data.__cat_dim__(key, surfactant_item)] = slice(surfactant_slices[idx], surfactant_slices[idx + 1])
            surfactant_data[key] = surfactant_item[s]
        
        # Create oil1 data
        for key in self.oil1_data.keys():
            oil1_item, oil1_slices = self.oil1_data[key], self.oil1_slices[key]
            s = list(repeat(slice(None), oil1_item.dim()))
            s[oil1_data.__cat_dim__(key, oil1_item)] = slice(oil1_slices[idx], oil1_slices[idx + 1])
            oil1_data[key] = oil1_item[s]
        
        # # Create oil2 data
        # for key in self.oil2_data.keys():
        #     oil2_item, oil2_slices = self.oil2_data[key], self.oil2_slices[key]
        #     s = list(repeat(slice(None), oil2_item.dim()))
        #     s[oil2_data.__cat_dim__(key, oil2_item)] = slice(oil2_slices[idx], oil2_slices[idx + 1])
        #     oil2_data[key] = oil2_item[s]

        # return api_data, cosolvent_data, surfactant_data, oil1_data, oil2_data
        return api_data, cosolvent_data, surfactant_data, oil1_data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        # Load in dataset in CSV format
        raw_path = 'data/training/raw/sedds.csv'
        smiles_list, mol_list, labels, ratios, properties = _load_dataset(raw_path)
        
        # Create data for API molecule
        data_list = []
        # cosolvent_data, surfactant_data, oil1_data, oil2_data = [], [], [], []
        # other_data_list = [cosolvent_data, surfactant_data, oil1_data, oil2_data]
        cosolvent_data, surfactant_data, oil1_data = [], [], []
        other_data_list = [cosolvent_data, surfactant_data, oil1_data]

        for i in range(len(smiles_list[0])):
            # Create molecular data and add labels
            data = mol_to_graph_data_obj_simple(mol_list[0][i], smiles_list[0][i])
            data.id = torch.tensor([i]) 
            data.y = torch.tensor(labels[i, :])
            data.ratios = torch.tensor(ratios[i, :])
            data.properties = torch.tensor(properties[i, :])
            data_list.append(data)

            # Create data for other chemicals (if they exist)
            for j in range(1, len(smiles_list)):
                try:
                    other_data = mol_to_graph_data_obj_simple(
                        mol_list[j][i], 
                        smiles_list[j][i]
                    )
                except:
                    other_data = Data(
                        x=torch.tensor([[0,0,0,0]], dtype=torch.long), 
                        edge_index=torch.tensor([[0],[0]], dtype=torch.long), 
                        edge_attr=torch.tensor([[0,0,0,0]], dtype=torch.long)
                    )
                other_data.id = torch.tensor([i]) 
                other_data_list[j-1].append(other_data)

        # Create batch subgraphs then save geometric processed data for drug
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        # Create batch subgraphs then save geometric processed data for other chemicals
        other_processed_paths = ['cosolvent', 'surfactant', 'oil']
        for i in range(len(other_data_list)):
            other_data, other_slices = self.collate(other_data_list[i])
            torch.save((other_data, other_slices), 'data/training/processed/geometric_data_processed_{}.pt'.format(other_processed_paths[i]))


# Run this to create the finetune dataset
if __name__ == "__main__":
    root = "data/training"
    os.makedirs(root + "/processed", exist_ok=True)
    dataset = MoleculeDataset(root)
