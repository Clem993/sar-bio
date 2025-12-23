# Tahoe-100M + GOSTAR Interactive Demo

A prototype demonstrating how transcriptomic drug response data becomes more interpretable when paired with curated compound and target intelligence from GOSTAR.

## Brand Assets

To use the official Excelra brand assets (logo, wave graphics, slogan), place the following files in the `assets/` folder:

- `Logo.png` — Primary Excelra logo
- `Wave.png` — Primary wave graphic
- `Wave2.png` — Alternate wave graphic
- `Slogan.png` — "Where data means more" tagline

The demo will automatically detect and use these assets if present. Without them, it will use text-based alternatives with the correct brand colours.

## Overview

This interactive demo connects:
- **Tahoe-100M** perturbation data (100M single-cell transcriptomic profiles across 1,100 compounds and 50 cell lines)
- **GOSTAR** structure-activity relationship data (10.6M+ compounds with curated SAR)

## Screens

1. **Overview** - Dataset statistics and integration concept
2. **Compound-Cell Response Network** - Network graph revealing which compounds produce similar cellular responses
3. **MoA Landscape** - 3D embedding of compounds positioned by transcriptomic response similarity
4. **Drug Response Trajectory** - Sankey diagram from compound/target through pathways to phenotype
5. **Cell Line Sensitivity Radar** - Radar charts comparing response profiles across cell lines
6. **Activity Cliff Detection** - Comparison of structurally similar compounds with divergent responses
7. **Target Deconvolution Dashboard** - Signature-matching to infer target engagement

## Running the Demo

### Prerequisites

```bash
pip install -r requirements.txt
```

### Launch

```bash
streamlit run app.py
```

The demo will open in your browser at `http://localhost:8501`

### For External Access

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Data

This prototype uses representative sample data for demonstration purposes. The data generation module (`data/sample_data.py`) creates:

- 100 mock compounds with GOSTAR-like annotations
- 20 cancer cell lines
- Response matrices and embeddings
- Differential expression and pathway enrichment data

To integrate real data:
1. Replace the sample data generator with actual Tahoe-100M data from HuggingFace
2. Connect GOSTAR API for compound lookups
3. Pre-compute embeddings using real transcriptomic data

## Customisation

### Styling
CSS is embedded in `app.py` - modify the `st.markdown` block with `<style>` tags.

### Data
Modify `data/sample_data.py` to change:
- Number of compounds
- Cell lines included
- Target class distributions
- Response characteristics

## For Production

To deploy this demo for sales use:

1. **Streamlit Cloud** (easiest): Push to GitHub, deploy via streamlit.io
2. **Internal hosting**: Docker container or VM with Streamlit server
3. **Embedded**: Use Streamlit's iframe embedding for integration with existing platforms

## Files

```
tahoe_demo/
├── app.py                 # Main Streamlit application
├── data/
│   ├── __init__.py
│   └── sample_data.py     # Mock data generation
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Next Steps

1. Integrate real Tahoe-100M subset data
2. Connect GOSTAR API for live compound lookups
3. Add structure visualisation (RDKit)
4. Implement export functionality (PDF, PNG)
5. Add authentication for external demos

---

*Demo version - Excelra Client Advisory & Consulting*
