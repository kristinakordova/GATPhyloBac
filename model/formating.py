import pandas as pd
import numpy as np

class PhenotypeProcessor:
    def __init__(self, phenotype_path, kleborate_path, antibiotic, BAPS):
        """
        Initialize with paths to the raw data files.
        """
        self.phenotype_path = phenotype_path
        self.kleborate_path = kleborate_path
        self.antibiotic = antibiotic
        self.BAPS = BAPS

    def format_phenotype(self, species_filter="Klebsiella pneumoniae"):
        """
        Cleans and merges phenotype and Kleborate data.
        """
        # Load data
        amr = pd.read_csv(self.phenotype_path)
        kleb = pd.read_csv(self.kleborate_path, delimiter="\t")
        # Filter Kleborate
        kleb = kleb[kleb["species"] == species_filter].copy()
        kleb["strain"] = kleb["strain"].astype(str)
        amr["Genome ID"] = amr["Genome ID"].astype(str)
        amr = amr[amr["Genome ID"].isin(kleb["strain"])].copy()
        # Clean extensions from Genome ID
        amr["Genome ID"] = amr["Genome ID"].str.replace(r'\.(fa|fna|fasta)$', '', regex=True)
        # Binary encoding
        amr["Resistant Phenotype"] = np.where(amr["Resistant Phenotype"] == 'Susceptible', 0, 1)
        amr = amr[amr["Antibiotic"]==self.antibiotic]
        amr = amr.drop_duplicates(subset='Genome ID')
        
        #add BAPS data
        Baps_file = pd.read_csv(self.BAPS)
        Baps_file.columns = ["Genome ID", "Baps"]
        Baps_file['Genome ID'] = Baps_file['Genome ID'].str.rstrip('.fna')
        amr = pd.merge(Baps_file, amr, on="Genome ID", how = "inner")

        return amr
        
    
class GraphProcessor:
    def __init__(self, file_path, amr_file):
        """
        Initialize with paths to the raw data files.
        """
        self.file_path = file_path
        self.amr_file = amr_file
        self.filtered_df = None 
        self.wide_df = None

    def load_and_filter_jaccard(self):
        """
        Loads Jaccard similarity, converts to distance, cleans extensions,
        and filters by valid genome IDs.
        """
        df = pd.read_csv(self.file_path, sep=" ", header=None)
        df[2] = 1 - df[2]
        
        #file extensions
        extension_pattern = r'\.(fa|fna|fasta)$'
        df[0] = df[0].str.replace(extension_pattern, '', regex=True)
        df[1] = df[1].str.replace(extension_pattern, '', regex=True)
        
        #Filter to only include pairs where BOTH samples are in valid_ids
        valid_ids = self.amr_file["index"].unique()
        mask = df.iloc[:, 0].isin(valid_ids) & df.iloc[:, 1].isin(valid_ids)
        df_filtered = df[mask].copy()
        
        #Assert check
        valid_set = set(valid_ids)
        found_samples = set(df_filtered.iloc[:, 0]).union(set(df_filtered.iloc[:, 1]))
        
        assert found_samples.issubset(valid_set), (
            f"Filtering Error: {len(found_samples - valid_set)} samples found in "
            f"filtered matrix that do not exist in clean_df."
        )
        self.filtered_df = df_filtered
        return df_filtered
    
    
    def cast_wide_matrix(self):
        """
        Ensures a perfectly symmetric square matrix with no mismatches.
        """
        # Strip whitespace
        self.filtered_df[0] = self.filtered_df[0].astype(str).str.strip()
        self.filtered_df[1] = self.filtered_df[1].astype(str).str.strip()


        all_samples = sorted(list(set(self.filtered_df[0]) | set(self.filtered_df[1])))
        wide_df = self.filtered_df.pivot(index=0, columns=1, values=2)
        wide_df = wide_df.reindex(index=all_samples, columns=all_samples)
        wide_df = wide_df.fillna(wide_df.T)

        #Set diagonal to 0
        np.fill_diagonal(wide_df.values, 0)
        return wide_df
    

class GraphConstructor:
    def __init__(self, wide_matrix):
        """
        Initialize with paths to the raw data files.
        """
        self.wide_matrix = wide_matrix
        self.graph_base = None
        self.distances = None
        self.indices = None
        self.G = None
        self.replicating_features_frame  = None
        self.graph_data = None
        self.edge_index_A = None

    def cmdscale(self, D):
        """Classical multidimensional scaling (MDS)
        """
        n = len(D)
        H = np.eye(n) - np.ones((n, n))/n
        B = -H.dot(D**2).dot(H)/2
        evals, evecs = np.linalg.eigh(B)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        w, = np.where(evals > 0)
        L = np.diag(np.sqrt(evals[w]))
        V = evecs[:, w]
        Y = V.dot(L)
        return Y, evals[evals > 0]
    
    def run_cmd(self, n_components = 30, target_variance = 90, max_dim = 200):
        import numpy as np


        #mds
        projection, evals = self.cmdscale(self.wide_matrix)
        
        #ummulative var
        total_variance = np.sum(evals)
        cumulative_variance = np.cumsum(evals) / total_variance
        
        #Print variance for specific checkpoints
        checkpoints = [30, 40, 50, 70, 90, 100, 120,150]
        print("--- MDS Variance Analysis ---")
        for cp in checkpoints:
            if cp <= len(cumulative_variance):
                var = cumulative_variance[cp-1] * 100
                print(f"Dimensions: {cp} | Explained Variance: {var:.2f}%")
        

        suggested_dim = np.where(cumulative_variance >= target_variance)[0]
        
        if len(suggested_dim) > 0:
            n_components = suggested_dim[0] + 1 # +1 for 0-based indexing
            print(f"Target of {target_variance*100}% reached at {n_components} dimensions.")
        else:
            n_components = max_dim
            print(f"Target variance not reached within limits. Using max_dim: {max_dim}")

        n_components = min(n_components, max_dim)
        
        # 5. Finalize projection
        graph_base = pd.DataFrame(projection)
        self.graph_base = graph_base.iloc[:, :n_components]
        
        print(f"Final Graph Base shape: {self.graph_base.shape}")
        return self.graph_base
    
    def run_knn(self, k=1):
        from sklearn.neighbors import kneighbors_graph
        import pandas as pd
        import networkx as nx
        import torch

        A = kneighbors_graph(self.graph_base, k)
        A_graph = nx.from_scipy_sparse_array(A)
        adj_A = nx.to_scipy_sparse_array(A_graph).tocoo()
        row_A = torch.from_numpy(adj_A.row.astype(np.int64)).to(torch.long)
        col_A = torch.from_numpy(adj_A.col.astype(np.int64)).to(torch.long)
        edge_index_A = torch.stack([row_A, col_A], dim=0)
        self.edge_index_A = edge_index_A
        return A_graph, edge_index_A

'''    def make_graph(self):
        import networkx as nx
        G = nx.Graph()
        #Map indices back to Sample IDs and add edges
        for i, row in enumerate(self.indices):
            sample_id = self.pca_df.index[i]
            neighbor_id = self.pca_df.index[row[1]]
            dist = self.distances[i, 1]
            # Add edge with the Jaccard/PCA distance as a weight
            G.add_edge(sample_id, neighbor_id, weight=dist)
        self.G = G
        return G
    '''

'''def get_features(self, DBGWAS_FEATURES, FEATURES, amr_pheno, test_cluster):
        with open(FEATURES, "r") as f:
            #get fearures for the respecive test cluster being excludedcluster
            for line in f:
                parts = line.strip().split(",")
                cluster_sub = int(parts[0].split("_")[1])
                if cluster_sub == test_cluster:
                    features_to_read = parts[1:]
                    features_as_ints = [int(f) for f in features_to_read]

        #get samples
        samples = amr_pheno["Genome ID"].to_list()
        samples = [f"{s}.fna" for s in samples]
        samples.append("ps")

        #ensure all samples are in the features file 
        header_df = pd.read_csv(
            DBGWAS_FEATURES, 
            sep=' ', 
            nrows=0
        )
        actual_columns = set(header_df.columns)
        required_samples = set(samples)
        missing_samples = required_samples - actual_columns
        samples = [s for s in samples if s not in missing_samples]

        #get the first 50
        features_as_ints = features_as_ints[0:50]
        #read_features
        replicating_features =  [int(feature) for feature in features_as_ints]
        all_features = set(range(0,5458393))
        rows_to_skip = all_features - set(replicating_features)
        rows_to_skip = rows_to_skip 
        rows_to_skip = list(rows_to_skip)
        rows_to_skip = [x+1 for x in rows_to_skip]          
        replicating_features_frame = pd.read_csv(DBGWAS_FEATURES, sep=' ',  skiprows = rows_to_skip, usecols= samples, index_col= "ps")
        replicating_features_frame.columns = replicating_features_frame.columns.str.replace(r'\.(fa|fna|fasta)$', '', regex=True)
        self.replicating_features_frame = replicating_features_frame
        return (replicating_features_frame)'''
    
''' def structure_graph_for_GAT(self, amr_pheno):
        import torch
        from torch_geometric.data import Data
        import numpy as np
        node_list = list(self.G.nodes())
        aligned_features = self.replicating_features_frame[node_list]
        x = torch.tensor(aligned_features.T.values, dtype=torch.float)

        # Create a mapping from Node ID to an integer index
        mapping = {node: i for i, node in enumerate(node_list)}
        edge_coords = []
        for u, v in self.G.edges():
            edge_coords.append([mapping[u], mapping[v]])
            edge_coords.append([mapping[v], mapping[u]]) 

        edge_index = torch.tensor(edge_coords, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index)
        amr_pheno_sub = amr_pheno.set_index("Genome ID")
        labels = torch.tensor(amr_pheno_sub.loc[node_list, 'Resistant Phenotype'].values, dtype=torch.float)
        data.y = labels'''
    

def get_freatures(FEATURES, cluster, DBGWAS_FEATURES, amr_pheno):
    import pandas as pd
    """
        Gets features from the unitig matrix for each cluster.
    """
    with open(FEATURES, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            cluster_found = int(parts[0].split("_")[1])
            if cluster_found == cluster: # get features based on requested cluster
                features_to_read = parts[1:]
                features_as_ints = [int(f) for f in features_to_read]
                print(f"found {len(features_as_ints)} features for cluster {cluster}")


    #get samples
    samples = amr_pheno["Genome ID"].to_list()
    samples = [f"{s}.fna" for s in samples]
    samples.append("ps")

    #ensure all samples are in the features file 
    header_df = pd.read_csv(
        DBGWAS_FEATURES, 
        sep=' ', 
        nrows=0
    )

    actual_columns = set(header_df.columns)
    required_samples = set(samples)
    missing_samples = required_samples - actual_columns
    samples = [s for s in samples if s not in missing_samples]

    #get the first 50. Remove this in actual analysis
    features_as_ints = features_as_ints[0:300000] #remove this later
    #read_features
    replicating_features =  [int(feature) for feature in features_as_ints]
    all_features = set(range(0,5458393))
    rows_to_skip = all_features - set(replicating_features)
    rows_to_skip = rows_to_skip 
    rows_to_skip = list(rows_to_skip)
    rows_to_skip = [x+1 for x in rows_to_skip]          
    replicating_features_frame = pd.read_csv(DBGWAS_FEATURES, sep=' ',  skiprows = rows_to_skip, usecols= samples, index_col= "ps")
    replicating_features_frame.columns = replicating_features_frame.columns.str.replace(r'\.(fa|fna|fasta)$', '', regex=True)
    replicating_features_frame = replicating_features_frame.T
    replicating_features_frame = replicating_features_frame.loc[:, (replicating_features_frame != 0).any(axis=0)]
    feature_names = list(replicating_features_frame.index)

    return replicating_features_frame,samples,feature_names 

























