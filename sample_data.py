"""
Sample data generation for Tahoe-100M + GOSTAR demo prototype.
This creates representative mock data for demonstration purposes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

np.random.seed(42)

# Cell lines from Tahoe-100M (representative subset)
CELL_LINES = [
    "A549", "MCF7", "HCT116", "PC3", "HeLa", "U2OS", "A375", "K562",
    "SKBR3", "MDA-MB-231", "HT29", "SW480", "PANC1", "BT474", "T47D",
    "COLO205", "NCI-H460", "OVCAR3", "DU145", "SKOV3"
]

# Target classes and mechanisms
TARGET_CLASSES = {
    "Kinase Inhibitor": ["EGFR", "BRAF", "MEK1/2", "PI3K", "mTOR", "CDK4/6", "JAK1/2", "ALK", "BTK", "FLT3"],
    "HDAC Inhibitor": ["HDAC1", "HDAC2", "HDAC3", "HDAC6", "Pan-HDAC"],
    "Proteasome Inhibitor": ["PSMB5", "PSMB1", "Proteasome 20S"],
    "Topoisomerase Inhibitor": ["TOP1", "TOP2A", "TOP2B"],
    "DNA Damage": ["PARP1", "ATR", "ATM", "DNA-PK", "CHK1"],
    "Apoptosis": ["BCL2", "BCL-XL", "MCL1", "XIAP"],
    "Epigenetic": ["BET", "EZH2", "DOT1L", "LSD1", "DNMT"],
    "Cell Cycle": ["PLK1", "Aurora A", "Aurora B", "WEE1", "TTK"],
    "Receptor Tyrosine Kinase": ["HER2", "VEGFR", "FGFR", "MET", "RET"],
    "Hormone": ["AR", "ER", "PR", "GR"]
}

# Sample compounds (representative of Tahoe-100M's 1,100 compounds)
def generate_compounds(n: int = 100) -> pd.DataFrame:
    """Generate mock compound data with GOSTAR-like annotations."""
    compounds = []
    
    compound_prefixes = ["EXC", "CPD", "TAH", "MOL", "DRG"]
    
    for i in range(n):
        target_class = np.random.choice(list(TARGET_CLASSES.keys()))
        primary_target = np.random.choice(TARGET_CLASSES[target_class])
        
        compound = {
            "compound_id": f"{np.random.choice(compound_prefixes)}-{1000 + i:04d}",
            "compound_name": f"Compound_{i+1}",
            "smiles": f"CC(C)Nc1ncnc2c1c(cn2C)c3ccc(cc3){'O' * np.random.randint(1, 4)}",  # Simplified mock SMILES
            "molecular_weight": round(np.random.uniform(250, 650), 1),
            "logP": round(np.random.uniform(1.0, 5.5), 2),
            "target_class": target_class,
            "primary_target": primary_target,
            "ic50_nm": round(10 ** np.random.uniform(0, 4), 1),  # 1 nM to 10 µM
            "selectivity_score": round(np.random.uniform(0.3, 1.0), 2),
            "clinical_stage": np.random.choice(
                ["Preclinical", "Phase I", "Phase II", "Phase III", "Approved"],
                p=[0.5, 0.2, 0.15, 0.1, 0.05]
            ),
            "gostar_match": np.random.choice([True, False], p=[0.7, 0.3]),
            "sar_datapoints": np.random.randint(5, 500) if np.random.random() > 0.3 else 0
        }
        compounds.append(compound)
    
    return pd.DataFrame(compounds)


def generate_response_matrix(compounds_df: pd.DataFrame, cell_lines: List[str]) -> pd.DataFrame:
    """Generate mock drug response matrix (compounds x cell lines)."""
    n_compounds = len(compounds_df)
    n_cell_lines = len(cell_lines)
    
    # Base response influenced by target class
    target_class_effects = {
        "Kinase Inhibitor": 0.6,
        "HDAC Inhibitor": 0.5,
        "Proteasome Inhibitor": 0.7,
        "Topoisomerase Inhibitor": 0.65,
        "DNA Damage": 0.55,
        "Apoptosis": 0.5,
        "Epigenetic": 0.45,
        "Cell Cycle": 0.6,
        "Receptor Tyrosine Kinase": 0.55,
        "Hormone": 0.4
    }
    
    response_matrix = np.zeros((n_compounds, n_cell_lines))
    
    for i, row in compounds_df.iterrows():
        base_effect = target_class_effects.get(row["target_class"], 0.5)
        potency_factor = 1 - (np.log10(row["ic50_nm"] + 1) / 4)  # More potent = stronger response
        
        for j in range(n_cell_lines):
            # Add cell line-specific variation
            cell_sensitivity = np.random.uniform(0.5, 1.5)
            noise = np.random.normal(0, 0.1)
            
            response = base_effect * potency_factor * cell_sensitivity + noise
            response_matrix[i, j] = np.clip(response, 0, 1)
    
    return pd.DataFrame(
        response_matrix,
        index=compounds_df["compound_id"],
        columns=cell_lines
    )


def generate_differential_expression(n_genes: int = 500) -> pd.DataFrame:
    """Generate mock differential expression data for selected compound-cell line pairs."""
    gene_symbols = [f"GENE{i}" for i in range(1, n_genes + 1)]
    
    # Add some real gene names for realism
    real_genes = [
        "TP53", "EGFR", "KRAS", "BRAF", "PIK3CA", "MYC", "PTEN", "RB1",
        "CDKN2A", "APC", "BRCA1", "BRCA2", "BCL2", "BAX", "CASP3", "CASP9",
        "MDM2", "STAT3", "JAK2", "MAPK1", "AKT1", "MTOR", "VEGFA", "HIF1A"
    ]
    gene_symbols[:len(real_genes)] = real_genes
    
    data = {
        "gene_symbol": gene_symbols,
        "log2fc": np.random.normal(0, 1.5, n_genes),
        "pvalue": 10 ** (-np.random.uniform(0, 10, n_genes)),
        "padj": 10 ** (-np.random.uniform(0, 8, n_genes)),
        "base_mean": 10 ** np.random.uniform(1, 4, n_genes)
    }
    
    df = pd.DataFrame(data)
    df["significant"] = (df["padj"] < 0.05) & (np.abs(df["log2fc"]) > 1)
    
    return df


def generate_pathway_enrichment() -> pd.DataFrame:
    """Generate mock pathway enrichment results."""
    pathways = [
        "Cell Cycle", "Apoptosis", "DNA Repair", "PI3K-AKT Signaling",
        "MAPK Signaling", "p53 Signaling", "mTOR Signaling", "Wnt Signaling",
        "Notch Signaling", "JAK-STAT Signaling", "NF-kB Signaling",
        "Oxidative Phosphorylation", "Glycolysis", "Autophagy",
        "Senescence", "EMT", "Hypoxia Response", "Inflammatory Response"
    ]
    
    data = {
        "pathway": pathways,
        "enrichment_score": np.random.uniform(-2, 2, len(pathways)),
        "pvalue": 10 ** (-np.random.uniform(0, 6, len(pathways))),
        "n_genes": np.random.randint(10, 200, len(pathways)),
        "direction": np.random.choice(["up", "down"], len(pathways))
    }
    
    return pd.DataFrame(data)


def generate_umap_coordinates(compounds_df: pd.DataFrame) -> pd.DataFrame:
    """Generate mock UMAP coordinates for compound response space."""
    n = len(compounds_df)
    
    # Create clusters based on target class
    target_classes = compounds_df["target_class"].unique()
    cluster_centers = {tc: (np.random.uniform(-5, 5), np.random.uniform(-5, 5)) 
                       for tc in target_classes}
    
    umap_coords = []
    for _, row in compounds_df.iterrows():
        center = cluster_centers[row["target_class"]]
        x = center[0] + np.random.normal(0, 1.5)
        y = center[1] + np.random.normal(0, 1.5)
        umap_coords.append({"compound_id": row["compound_id"], "umap_1": x, "umap_2": y})
    
    return pd.DataFrame(umap_coords)


def generate_3d_embedding(compounds_df: pd.DataFrame) -> pd.DataFrame:
    """Generate mock 3D embedding for MoA landscape."""
    n = len(compounds_df)
    
    # Create clusters based on target class
    target_classes = compounds_df["target_class"].unique()
    cluster_centers = {tc: (np.random.uniform(-5, 5), 
                            np.random.uniform(-5, 5),
                            np.random.uniform(-5, 5)) 
                       for tc in target_classes}
    
    coords = []
    for _, row in compounds_df.iterrows():
        center = cluster_centers[row["target_class"]]
        x = center[0] + np.random.normal(0, 1.2)
        y = center[1] + np.random.normal(0, 1.2)
        z = center[2] + np.random.normal(0, 1.2)
        coords.append({
            "compound_id": row["compound_id"],
            "x": x, "y": y, "z": z,
            "target_class": row["target_class"],
            "primary_target": row["primary_target"]
        })
    
    return pd.DataFrame(coords)


def get_sample_data() -> Dict:
    """Generate all sample data for the demo."""
    compounds = generate_compounds(100)
    cell_lines = CELL_LINES
    response_matrix = generate_response_matrix(compounds, cell_lines)
    umap_coords = generate_umap_coordinates(compounds)
    embedding_3d = generate_3d_embedding(compounds)
    diff_expr = generate_differential_expression()
    pathways = generate_pathway_enrichment()
    
    return {
        "compounds": compounds,
        "cell_lines": cell_lines,
        "response_matrix": response_matrix,
        "umap_coords": umap_coords,
        "embedding_3d": embedding_3d,
        "differential_expression": diff_expr,
        "pathways": pathways,
        "target_classes": TARGET_CLASSES
    }


if __name__ == "__main__":
    data = get_sample_data()
    print(f"Generated {len(data['compounds'])} compounds")
    print(f"Response matrix shape: {data['response_matrix'].shape}")
