#!/usr/bin/python

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, PandasTools, Descriptors, rdmolops
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

def compute_tanimoto_distance_matrix(fingerprint_list):
    """
    Compute the Tanimoto distance matrix for a list of fingerprints.
    
    Args:
        fingerprint_list (list): List of fingerprints.
        
    Returns:
        list: Tanimoto distance matrix.
    """
    distance_matrix = []
    for i in range(1, len(fingerprint_list)):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint_list[i], fingerprint_list[:i])
        for x in similarities:
                distance_matrix.append(1 - x)
    return distance_matrix

def cluster_fingerprints(fingerprints, threshold):
    """
    Cluster fingerprints using the Butina algorithm.
    
    Args:
        fingerprints (list): List of fingerprints.
        threshold (float): Cutoff value for clustering.
        
    Returns:
        list: Clusters of fingerprints.
    """
    distance_matrix = compute_tanimoto_distance_matrix(fingerprints)
    clusters = Butina.ClusterData(distance_matrix, len(fingerprints), threshold, isDistData=True)
    return clusters

def main():
    """
    Main function to read input data, calculate fingerprints, and perform clustering.
    """
    if len(sys.argv) != 3:
        print("Usage: python ligand_clustering.py <input_file> cutoff")
        sys.exit(1)
    
    input_file = sys.argv[1]
    cutoff = float(sys.argv[2])
    if not os.path.exists(input_file):
        print("Error: File '{input_file}' not found.")
        sys.exit(1)
    
    data_frame = pd.read_csv(input_file, delim_whitespace=True, header=None, names=["ID", "Smiles","Dock_score"])

# Reading the input
    molecules = []
    for i in data_frame.index:
        Id = data_frame['ID'][i]
        smiles = data_frame['Smiles'][i]
        molecules.append((Chem.MolFromSmiles(smiles), Id))

# Generating ECFP4 fingerprints of compounds
    rdkit_fp_generation = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    fingerprints = [ rdkit_fp_generation.GetFingerprint(m) for m,Id in molecules ]
 
    clusters = cluster_fingerprints(fingerprints,cutoff)

# OUTPUT cluster information using Chimera format
    print(len(fingerprints),'compounds converted successfully!')
    list_out = []
    for n in range(len(clusters)):
        for j in range(len(clusters[n])):
            member_idx = clusters[n][j]
            zinc_id = molecules[member_idx][1]                                                   
            cluster_id    = f'cluster_{n+1:04d}'
            cluster_label = f'c_{n+1:04d}_{j+1:03d}'
            list_out.append([zinc_id,cluster_id,cluster_label])

    data_frame_cluster = pd.DataFrame(list_out, columns=["ID","cluster","label"])
    data_frame_cluster.to_csv("output.csv", index=False)

if __name__ == "__main__":
    main()

