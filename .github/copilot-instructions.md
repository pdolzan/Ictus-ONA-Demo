# Ictus ONA Demo - AI Coding Agent Instructions

## Project Overview
This is a **Streamlit-based interactive web application** demonstrating Organizational Network Analysis (ONA) using fictional movie characters. The app teaches how informal relationships (trust, advice-seeking, friendship) shape organizational dynamics, using a gamified scenario where users map their connections within a "fellowship of heroes."

**Key Domain**: Organizational Network Analysis (ONA) / Organizational Psychology  
**Tech Stack**: Python + Streamlit + NetworkX + Plotly + pyvis + igraph  
**Language**: Slovenian UI content

---

## Architecture & Data Flow

### Core Pattern: Session-Based Network Modeling
The app uses **Streamlit session state** to maintain user selections across reruns:

1. **User Input Tab (Tab 2)**: Collects three relationship types via checkboxes
   - Trust selections → stored in `st.session_state["trust"]`
   - Advice selections → stored in `st.session_state["advice"]`
   - Friend selections → stored in `st.session_state["friends"]`

2. **Matrix Generation Tab (Tab 3)**: Converts selections into adjacency matrices
   - Calls `update_matrices_if_needed()` which caches matrices using `st.session_state`
   - **Critical**: Matrices persist across tab switches via session state keys like `trust_matrix_others`
   - Only regenerates if user changes selections or key missing from session

3. **Network Visualization (Tab 4 & 7)**: Converts matrices to graph visualizations
   - Uses **igraph** for centrality metrics (betweenness, closeness, pagerank, clustering, coreness)
   - Uses **NetworkX** for layout algorithms and additional metrics
   - Uses **Force Atlas 2** (`fa2_modified`) for graph positioning

### Key Data Structure: Character List
```python
characters = [
    "Deček čarovnik", "Pametna čarovnica", ..., "Škratovski bojevnik"  # 18 total
]
all_names = ["TU STE VI"] + characters  # "TU STE VI" = user ("You Are Here"), always index 0
```
The user is always row/column 0. This matters for matrix operations—**never modify row 0 after user input**.

---

## Critical Implementation Patterns

### Pattern 1: Session State Management for Matrix Persistence
**Location**: Lines 370-390 (`update_matrices_if_needed()`)

Matrices are **expensive to compute** (random generation + reciprocity logic). They're cached using:
```python
if matrix_key not in st.session_state:
    mat, others_mtx = create_matrix(...)
    st.session_state[matrix_key] = mat
    st.session_state[others_key] = others_mtx  # <- Store the random choices separately
```

**Why split storage?** The `others_matrix` stores NPC choices, allowing user row to update independently without regenerating all random choices.

**When modifying**: Always preserve `st.session_state["*_matrix_others"]` keys.

### Pattern 2: Non-Symmetric Adjacency Matrices with User Row Protection
**Location**: Lines 310-345 (`create_matrix()`)

The user (row 0) can change selections freely, but NPC selections must stay consistent within a session:

```python
# User row: always regenerated from current selections
for selected_name in user_selection:
    j = all_names.index(selected_name)
    matrix[user_idx, j] = 1  # User is always row 0

# NPC rows: loaded from cache if it exists, else generated randomly
if others_matrix is None:
    for i in range(1, n):
        # Generate random choices for NPC i
    # Apply reciprocity (but NEVER modify row 0)
    for i in range(1, n):  # SKIP row 0
        if matrix[i, j] == 1 and matrix[j, i] == 0:
            if random.random() < reciprocity_rate:
                matrix[j, i] = 1
    # Re-freeze user row after reciprocity
    fixed_row = [...]
    matrix[user_idx, :] = fixed_row
```

**Key invariant**: Row 0 never gets overwritten after initial user selection.

### Pattern 3: Three Relationship Matrices with Configurable Parameters
**Location**: Lines 370-385 (matrix iteration loop)

Each relationship type has different **density** and **reciprocity** values:
```python
("trust_matrix",   "trust",    density=0.15, reciprocity=0.2),
("advice_matrix",  "advice",   density=0.15, reciprocity=0.2),
("friends_matrix", "friends",  density=0.08, reciprocity=0.3),
```

- **Lower density** = fewer edges (friends are more selective)
- **Higher reciprocity** = more two-way relationships (friends reciprocate more than advice)

Tweak these for different network behaviors.

### Pattern 4: Combined Matrix for Holistic Analysis
**Location**: Line 388

```python
st.session_state["combined_matrix"] = (
    st.session_state["trust_matrix"] +
    st.session_state["advice_matrix"] +
    st.session_state["friends_matrix"]
)
```
Matrix values become **1, 2, or 3** (overlap count). This reveals who appears in multiple relationship types.

---

## Visualization Pipelines

### Static Graph Visualization (Tab 4)
**Location**: Lines 550+ (`plot_static_network()`)

- **Input**: Adjacency matrix
- **Library**: igraph + matplotlib
- **Output**: SVG saved to disk, then rendered via `st.components.v1.html()`

**Node sizing logic**:
- Small nodes (size 150-950) proportional to weighted degree
- Special star marker for user ("TU STE VI")
- Square markers for formal leaders

**Node coloring**: Heatmap (light→dark) based on betweenness centrality.

**Edge rendering**: Arrows with weights 1/2/3 rendered in different grays; higher-weight edges drawn last.

### Interactive Graph Visualization (Tab 7)
**Location**: Lines 900-1003 (`interactive_network.html`)

- **Library**: pyvis (web-based network visualization)
- **Features**: 
  - User selects size/color metrics independently
  - Scenario analysis: remove nodes and observe network impact
  - Hover tooltips with all metrics
- **Styling**: Nodes with different shapes (star for user, square for leaders)

**Important**: The HTML file is **generated locally and persisted** as `interactive_network.html` in the working directory.

---

## Key Metrics Explained (for context)

These are computed and displayed; understanding them aids in feature additions:

| Metric | Meaning | Use Case |
|--------|---------|----------|
| **Betweenness** | How often someone bridges unconnected groups | Identifies information gatekeepers |
| **Degree** | Total connections (in + out) | Overall network activity |
| **Closeness** | Average distance to others | Speed of reaching network |
| **PageRank** | Importance via connected peers | Influence (like Google ranking) |
| **Clustering Coefficient** | Local cliquishness | Identifies tight subgroups |
| **Coreness** | Position in core vs. periphery | Integration level |

---

## Language & Localization

**All UI text is in Slovenian** (characters, labels, descriptions). When adding features:
- Keep strings in the code (not database)
- Maintain Slovenian for consistency with existing content
- Use UTF-8 encoding (file already uses it correctly)

**Character names** are hardcoded. Don't parameterize unless adding a new scenario.

---

## External Dependencies - Key Imports

| Module | Purpose | Notes |
|--------|---------|-------|
| `streamlit` | Web framework | V1.x; page_config at top of file |
| `networkx` | Graph algorithms | DiGraph (directed) used; layout metrics |
| `igraph` | Fast centrality | Used in Tab 4 for metric computation |
| `matplotlib` | Static visualization | SVG export; DejaVu Serif font enforced |
| `plotly` | Interactive heatmaps | Heatmap table generation |
| `pyvis` | Interactive networks | HTML export; used in Tab 7 |
| `fa2_modified` | Force-directed layout | Custom import (see `lib/` folder) |
| `pandas` | Data structures | Matrices represented as DataFrames |

---

## Common Dev Tasks

### Add a New Relationship Type
1. Add character selection in Tab 2 (expander + checkboxes)
2. Add session state initialization: `if key not in st.session_state: st.session_state[key] = []`
3. Add to matrix generation loop in `update_matrices_if_needed()` with density/reciprocity params
4. Update matrix display in Tab 3
5. Update matrix dict in Tabs 4 & 7

### Modify Network Layout
- Edit `ForceAtlas2` parameters in Tab 4 (lines ~470-480)
- Common tweaks: `gravity`, `scalingRatio`, `edgeWeightInfluence`

### Add/Remove Characters
- Modify `characters` list (top of file)
- Update info table in Tab 1 and Tab 7
- Hierarchy mermaid diagram in Tab 1 will need manual update

### Change Metrics Displayed
- Modify metric computation in `plot_static_network()` (igraph calls like `g.betweenness()`)
- Update metric selectbox options in Tab 7

---

## Testing & Debugging

**No automated tests exist.** Manual testing:
1. Run: `streamlit run IctusONA_Demo_Version2.py`
2. Check each tab for rendering issues
3. Verify matrix values stay consistent across reruns
4. Confirm user row (row 0) doesn't change when switching tabs

**Debug session state**: Add `st.write(st.session_state)` in any tab to inspect.

---

## File Organization

```
.
├── IctusONA_Demo_Version2.py      ← Main app (current version)
├── IctusONA_Demo.py               ← Older version (kept for reference)
├── requirements.txt               ← Dependencies
├── interactive_network.html       ← Generated; don't edit directly
├── lib/
│   ├── bindings/utils.js
│   ├── tom-select/                ← Not used in V2
│   └── vis-9.1.2/                 ← Not used in V2
├── images/
│   ├── icon.png
│   ├── img1.png, img2.png         ← Ictus company visuals (Tab 6)
└── .streamlit/config.toml         ← Streamlit configuration
```

**V2 is current**. V1 files kept for reference only.

---

## Known Quirks & Gotchas

1. **SVG Files Generated Per Tab**: Each graph in Tab 4 exports to an SVG file (e.g., `Zaupanje.svg`). These are **created in the working directory**, not in a temp folder. Clean up manually if needed.

2. **Font Dependencies**: DejaVu Serif is enforced for cross-platform consistency. If graphs render with wrong font, check system fonts.

3. **Streamlit Reruns**: Every interaction triggers a full rerun. Session state prevents redundant matrix regeneration, but understand the rerun cycle when debugging.

4. **Slovenian String Handling**: Some strings have smart quotes (e.g., `"Čaka vas nova pustolovščina!"`). Ensure UTF-8 is preserved when editing.

5. **igraph vs NetworkX**: Both are used for different purposes. igraph computes metrics faster; NetworkX provides graph objects. Don't assume equivalence.

---

## Quick Command Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run IctusONA_Demo_Version2.py

# Run with debug settings
streamlit run IctusONA_Demo_Version2.py --logger.level=debug
```

---

## Entry Points for New Features

- **Add relationship analysis**: Extend matrix types → Tab 3 / Tab 4 / Tab 7
- **Add educational content**: Expand Tab 5 (ONA Applications) table
- **Customize character roster**: Modify `characters` list + info table
- **Change network algorithms**: Swap layout engine or centrality metrics in `plot_static_network()`
- **Add export functionality**: Add download button for matrices/graphs (new in Tab)

---

**Last Updated**: November 2025  
**Codebase**: Single-file Streamlit app (1003 lines)  
**Audience**: ONA practitioners, organizational psychologists, educational users
