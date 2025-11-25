import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import random
from pyvis.network import Network
import matplotlib.colors as mcolors
import igraph as ig
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import streamlit.components.v1 as components
import hashlib
import html
import tempfile, os
from fa2_modified import ForceAtlas2
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import TextPath
import base64
import re
from pathlib import Path

ASSETS = Path("images")

st.set_page_config(page_title="ICTUS ONA Demo", layout="centered", page_icon=str(ASSETS/"icon.png"))
plt.rcParams["font.serif"] = ["DejaVu Serif"]  # cross-platform




st.markdown(
    """
    <h2 style='color:#fffff; font-size:40px;'>Čaka vas nova pustolovščina!</h2>
    </br>
    <p style='font-size:20px;'>
Svet je v nevarnosti in tokrat za rešitev potrebujemo sodelovanje junakov iz različnih svetov.
Nekaj podobnega kot združitev več podjetij, kjer mora vsaka ekipa uskladiti delovanje, da bo celota uspešna.
Vsak junak ima svoje sposobnosti, a za uspeh je ključnega pomena, kako se znajde v novem okolju in kulturi. 
V tem izzivu bo vsak od vas moral razumeti, se prilagoditi in sodelovati z različnimi ekipami, da 
bo druščina učinkovita. Ne pozabimo, da največja moč ekipe ni vedno formalna hierarhija. Pomembne so 
neformalne vezi, komu zaupate, koga bi prosili za nasvet in kdo je vaš zaveznik, ko naletite na izzive.
Zato vas sprašujemo: Komu zaupate, koga bi prosili za nasvet in kdo je vaš zaveznik v tej druščini junakov?
S tem bomo odkrili vašo pozicijo ter razumeli, v kakšno druščino se podajate. Vprašanje je ali bo prav vaša povezava z drugimi pripomogla k uspehu celotnega podviga.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr style='border: 1px dashed  #fac108; margin: 1rem 0;'>", unsafe_allow_html=True)
st.write("""
         *Ta preprosta demonstracija prikazuje, kako se uporablja Analiza organizacijskih omrežij (ONA)
tokrat s pomočjo znanih likov iz filmov. Izberite, komu bi zaupali, s kom bi sodelovali in kdo
bo vaš vir informacij. Rezultat je vizualna predstavitev omrežja, ki prikazuje, kako se lahko ONA
uporablja za razumevanje dinamike v skupinah in organizacijah. 
         (Ustvarjeno v Python-u s pomočjo ChatGPT-ja objavljeno na Streamlit-u)*
        """)
st.markdown("<hr style='border: 1px dashed   #fac108; margin: 1rem 0;'>", unsafe_allow_html=True)
st.write("")
st.write("")



# --- Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [ "Hierarhija","Vprašalnik", "Matrike", "Omrežje", "ONA Aplikacije","Ictus Vizitka", "Interactive"])

st.markdown(
    """
    <style>
    /* Tab label font style */
    button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #003A6E !important;  /* text color */
        margin-right: 3px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

characters = [
    "Deček čarovnik", "Pametna čarovnica", "Modri ravnatelj", "Mladi pilot",
    "Tihotapec", "Zeleni mojster", "Nosilec prstana", "Sivi čarovnik",
    "Postopač", "Vilinski lokostrelec", "Škratovski bojevnik", "Zvesti robot",
    "Voditeljica upora", "Temni vitez", "Zvesti prijatelj", "Zlati robot", "Prebrisani učitelj", "Pogumna princesa"
]

all_names = ["TU STE VI"] + characters

# --- TAB 1: Hierarhična struktura
with tab1:
    st.write("")
    st.write(
        "Vaša druščina junakov in njena hierarhična struktura odločanja."
    )

    # apply the compact table css in a scoped container and render the static md table
    from streamlit.components.v1 import html

    html("""
        <div id="hier-table">
        <style>
            /* Ensure the outer container is white */
            #hier-table {
                background-color: white !important;
                padding: 0;
            }

            /* Table styling */
            #hier-table table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 0px;
                background-color: white !important;
            }

            /* Header and cell styling */
            #hier-table th, #hier-table td {
                padding: 4px 6px;
                text-align: left;
                border-bottom: 1px solid #ddd;
                font-family: "DejaVu Serif", serif;
                font-size: 12px;
                line-height: 0.9;
                background-color: white !important;
                color: black !important;  /* ensures text is visible in dark mode */
            }

            #hier-table th {
                font-weight: 600;
            }
        </style>
        
         <table>
            <thead>
                <tr>
                    <th>Ime</th>
                    <th>Vloga</th>
                    <th># let v druščini</th>
                </tr>
            </thead>
            <tbody>
            <tr><td>Deček čarovnik</td><td>premaga vse izzive</td><td>3</td></tr>
            <tr><td>Mladi pilot</td><td>pogumni raziskovalec</td><td>1</td></tr>
            <tr><td>Modri ravnatelj</td><td>vodi druge da so boljši</td><td>10</td></tr>
            <tr><td>Nosilec prstana</td><td>odgovoren vendar zadržan</td><td>2</td></tr>
            <tr><td>Pametna čarovnica</td><td>vir znanja in logike</td><td>3</td></tr>
            <tr><td>Pogumna princesa</td><td>odločna in napredna</td><td>1</td></tr>
            <tr><td>Postopač</td><td>skriti vodja z notranjim bojem</td><td>9</td></tr>
            <tr><td>Prebrisani učitelj</td><td>povezovalec</td><td>12</td></tr>
            <tr><td>Sivi čarovnik</td><td>povezuje ljudi in jih vodi</td><td>30</td></tr>
            <tr><td>Temni vitez</td><td>konfliktna oseba</td><td>13</td></tr>
            <tr><td>Tihotapec</td><td>samosvoj</td><td>6</td></tr>
            <tr><td>Vilinski lokostrelec</td><td>spreten in zvest</td><td>12</td></tr>
            <tr><td>Voditeljica upora</td><td>pogumna in strateška voditeljica</td><td>2</td></tr>
            <tr><td>Zeleni mojster</td><td>učitelj in vir discipline</td><td>20</td></tr>
            <tr><td>Zlati robot</td><td>raztreseni in zvesti pomočnik</td><td>7</td></tr>
            <tr><td>Zvesti prijatelj</td><td>zvesti podpornik in srce ekipe</td><td>4</td></tr>
            <tr><td>Zvesti robot</td><td>iznajdljiv zvesti pomočnik</td><td>9</td></tr>
            <tr><td>Škratovski bojevnik</td><td>zvest in neustrašen</td><td>12</td></tr>
            </tbody>
        </table>
        </div>
        """, height=400)


    import streamlit.components.v1 as components

    mermaid_code = r"""
<div class="mermaid">
%%{init: {'flowchart': {'curve': 'linear'}} }%%
graph TD
    CEO["Sivi čarovnik"]
    CEO --> HR["Voditeljica upora"]
    CEO --> Sales["Modri ravnatelj"]
    CEO --> Event["Zeleni mojster"]
    HR --> D1["Deček čarovnik"]
    HR --> D2["Pametna čarovnica"]
    Sales --> S1["Zlati robot"]
    Sales --> S2["Mladi pilot"]
    Event --> E1["Tihotapec"]
    Event --> E2["Pogumna princesa"]
    EP2 --> X1["Nosilec prstana"]
    E1 --> EP2["Prebrisani učitelj"]
    E1 --> EP3["Postopač"]
    EP3 --> EP4["Vilinski lokostrelec"]
    E2 --> EP5["Temni vitez"]
    E2 --> EP6["Zvesti robot"]
    EP5 --> F1["Zvesti prijatelj"]
    EP5 --> F2["Škratovski bojevnik"]
    Event --> YOU["TU STE VI"]
    style YOU fill:#fac108
    </div>

<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({ startOnLoad: true, theme: 'neutral' });
</script>
"""
    components.html(mermaid_code, height=450)

# --- TAB 2: Vprašalnik
with tab2:
    st.write("")
    st.write(
        "Za vse junake smo pripravili sklop treh vprašanj, ki nam bodo v pomoč pri razumevanju kakšni so odnosi v vaši druščini. " \
        "Pri vsakem vprašanju je seznam vseh članov druščine.  Prosimo, da pri vsakem vprašanju izberete" \
        ", **komu zaupate**, **koga bi prosili za nasvet** in **kdo je vaš prijatelj**. "
        "Vsako vprašanje lahko odprete ali zaprete za lažji pregled." \
        
    )

    # Initialize session_state for storing selections
    for key in ["trust", "advice", "friends"]:
        if key not in st.session_state:
            st.session_state[key] = []

    # ---------- TRUST ----------
    with st.expander("1. Komu zaupate?", expanded=False):
        cols_trust = st.columns(3)
        trust_selection = []
        for i, name in enumerate(characters):
            col = cols_trust[i % 3]
            with col:
                checked = st.checkbox(
                    name,
                    key=f"trust_{i}",
                    value=(name in st.session_state["trust"])
                )
                if checked:
                    trust_selection.append(name)

        # Sync to session_state
        st.session_state["trust"] = trust_selection

    # ---------- ADVICE ----------
    with st.expander("2. Koga vse bi prosili za nasvet?", expanded=False):
        cols_advice = st.columns(3)
        advice_selection = []
        for i, name in enumerate(characters):
            col = cols_advice[i % 3]
            with col:
                checked = st.checkbox(
                    name,
                    key=f"advice_{i}",
                    value=(name in st.session_state["advice"])
                )
                if checked:
                    advice_selection.append(name)

        st.session_state["advice"] = advice_selection

    # ---------- FRIENDS ----------
    with st.expander("3. Kdo so vaši prijatelji?", expanded=False):
        cols_friends = st.columns(3)
        friends_selection = []
        for i, name in enumerate(characters):
            col = cols_friends[i % 3]
            with col:
                checked = st.checkbox(
                    name,
                    key=f"friends_{i}",
                    value=(name in st.session_state["friends"])
                )
                if checked:
                    friends_selection.append(name)

        st.session_state["friends"] = friends_selection

    # Optional: display current selections for debugging
    #with st.expander("🔍 Preglej trenutne izbire"):
    #    st.write("**Zaupanje:**", st.session_state["trust"])
    #    st.write("**Nasveti:**", st.session_state["advice"])
    #   st.write("**Prijatelji:**", st.session_state["friends"])

# --- FUNKCIJA ZA USTVARJANJE MATRICE (non-symmetrical) ---
def create_matrix(user_selection, density=0.15, reciprocity_rate=0.3):
    """
    Ustvari nesimetrično matriko, ki vsebuje uporabnikove izbire in
    naključne izbire ostalih likov, z možnostjo določene stopnje reciprocitete.
    
    Parameters:
        user_selection: list of characters selected by the user
        density: fraction of possible edges each character will have
        reciprocity_rate: fraction of edges that will be reciprocated
    """
    n = len(all_names)
    matrix = np.zeros((n, n), dtype=int)

    user_idx = 0  # TU STE VI

    # 1️⃣ User selections → freeze this row later
    for selected_name in user_selection:
        if selected_name in characters:
            j = all_names.index(selected_name)
            matrix[user_idx, j] = 1

    # 2️⃣ Random choices for OTHER characters (i > 0)
    for i in range(1, n):
        num_choices = max(1, int(density * (n - 1)))
        possible = [name for name in all_names if name != all_names[i]]
        choices = random.sample(possible, num_choices)
        for chosen in choices:
            j = all_names.index(chosen)
            matrix[i, j] = 1

    # 3️⃣ Reciprocity — DO NOT modify row 0
    for i in range(1, n):          # skip user row
        for j in range(n):
            if matrix[i, j] == 1 and matrix[j, i] == 0:
                # allowed: j == 0 (others → user)
                if random.random() < reciprocity_rate:
                    matrix[j, i] = 1

    # 4️⃣ Re-freeze the user row (protect user selections)
    fixed_row = np.zeros(n, dtype=int)
    for selected_name in user_selection:
        j = all_names.index(selected_name)
        fixed_row[j] = 1
    matrix[user_idx, :] = fixed_row

    return pd.DataFrame(matrix, index=all_names, columns=all_names)

# --- FUNKCIJA ZA USTVARJANJE TABELE ---
def plot_heatmap_table(matrix, title):
    import numpy as np

    z = matrix.values
    x = matrix.columns
    y = matrix.index

    # Create a new z_color array: 1 → colored, 0 → white
    z_color = np.where(z >= 1, 1, 0)  # 1 becomes colored, 0 stays white

    # Heatmap with custom colorscale: 0 = white, 1 = blue (or any color)
    fig = go.Figure(data=go.Heatmap(
        z=z_color,
        x=x,
        y=y,
        colorscale=[[0, 'white'], [1, "#E6E4E1"]],
        showscale=False,
        text=z,
        texttemplate="%{text}",
        textfont={"size":12},
        hoverinfo='none'
    ))

    # Add grid lines manually
    shapes = []
    n_rows, n_cols = z.shape
    for i in range(n_rows + 1):
        shapes.append(dict(
            type="line",
            x0=-0.5,
            x1=n_cols - 0.5,
            y0=i - 0.5,
            y1=i - 0.5,
            line=dict(color="black", width=1)
        ))
    for j in range(n_cols + 1):
        shapes.append(dict(
            type="line",
            x0=j - 0.5,
            x1=j - 0.5,
            y0=-0.5,
            y1=n_rows - 0.5,
            line=dict(color="black", width=1)
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-90, side='top', tickmode='array', tickvals=list(range(len(x))), ticktext=list(x)),
        yaxis=dict(autorange='reversed', tickmode='array', tickvals=list(range(len(y))), ticktext=list(y)),
        height=550,
        margin=dict(l=80, r=20, t=50, b=80),
        shapes=shapes
    )

    st.plotly_chart(fig, width='stretch')

# --- FUNKCIJA ZA USTVARJANJE MATRIKE LE ENKRAT NA SESION ---
def update_matrices_if_needed():
    # Check whether matrices already exist
    if "trust_matrix" not in st.session_state:
        st.session_state["trust_matrix"] = create_matrix(st.session_state["trust"], 0.15, 0.2)
        st.session_state["advice_matrix"] = create_matrix(st.session_state["advice"], 0.15, 0.2)
        st.session_state["friends_matrix"] = create_matrix(st.session_state["friends"], 0.08, 0.3)
        st.session_state["combined_matrix"] = (
            st.session_state["trust_matrix"] +
            st.session_state["advice_matrix"] +
            st.session_state["friends_matrix"]
        )
        return

    # Check if user changed selections
    if (
        st.session_state["trust"] != st.session_state.get("last_trust", None) or
        st.session_state["advice"] != st.session_state.get("last_advice", None) or
        st.session_state["friends"] != st.session_state.get("last_friends", None)
    ):
        # Update matrices
        st.session_state["trust_matrix"] = create_matrix(st.session_state["trust"], 0.15, 0.2)
        st.session_state["advice_matrix"] = create_matrix(st.session_state["advice"], 0.15, 0.2)
        st.session_state["friends_matrix"] = create_matrix(st.session_state["friends"], 0.08, 0.3)
        st.session_state["combined_matrix"] = (
            st.session_state["trust_matrix"] +
            st.session_state["advice_matrix"] +
            st.session_state["friends_matrix"]
        )

    # Save snapshot of current answers
    st.session_state["last_trust"] = st.session_state["trust"]
    st.session_state["last_advice"] = st.session_state["advice"]
    st.session_state["last_friends"] = st.session_state["friends"]

# --- TAB 3: Matrike
with tab3:
    st.write("")
    st.write("Na podlagi vaših odgovorov in odgovorov vaše druščine smo sestavili podatkovne matrike. "
             "V podatkovnih matrikah so povezave označene numerično z **1** ob ustreznem paru imen. Matrika ni simetrična "
             "saj tudi povezave niso vedno vzajemne. "
    "Vse predhodne matrike smo združili v zadnjo skupno matriko, ki nam bo služila za prikaz vrednostnega omrežja. "
   "V vrednostnem omrežju so povezave ovrednotene. Oseba, ki se pojavi v vseh treh matrikah je v naši druščini zelo pomembna. "
   "Saj je za ostale zaupanje vreden vir informacij in ga obravnavajo kot prijatelja.")

    if all(k in st.session_state for k in ["trust", "advice", "friends"]):

        # Ustvari 3 matrike z uporabnikovimi in naključnimi povezavami
        update_matrices_if_needed()

        # Use stored matrices
        trust_matrix = st.session_state["trust_matrix"]
        advice_matrix = st.session_state["advice_matrix"]
        friends_matrix = st.session_state["friends_matrix"]
        combined_matrix = st.session_state["combined_matrix"]


        # Display original matrices
        with st.expander("Matrika zaupanja", expanded=False):
            st.write("")
            plot_heatmap_table(trust_matrix,"")

        with st.expander("Matrika svetovanja", expanded=False):
            st.write("")
            plot_heatmap_table(advice_matrix, "")

        with st.expander("Matrika prijateljstva", expanded=False):
            st.write("")
            plot_heatmap_table(friends_matrix, "")

        # Combine matrices: element-wise sum
        with st.expander("Združena matrika", expanded=False):
            st.write("")
            plot_heatmap_table(combined_matrix, "")

    else:
        st.warning("Najprej izpolnite vprašalnik.")


# --- TAB 4 --- Grafi
with tab4: 
    st.write("")
    st.write(
        "Grafi prikazujejo, kdo v druščini junakov združuje skupine, kdo je v središču dogajanja in " \
        "kdo deluje kot most med različnimi ekipami. Velik krog pomeni, da ima junak veliko povezav, intenzivnost barve pa prikazuje, " \
        "kako pomemben je junak pri povezovanju različnih delov omrežja. Opazujemo, kdo je izoliran, kdo pospešuje sodelovanje "
        "in kako močno je omrežje kot celota. S pomočjo gostote omrežja spremljamo, kako tesno " \
        "so povezani junaki med seboj, medtem ko recipročnost povezav pokaže, koliko je " \
        "sodelovanje obojestransko. To nam pomaga razumeti, kdo so vplivneži in katalizatorji. Junaki, " \
        "ki morda niso formalno vodilni ali najbolj glasni, a znajo posredovati informacije in ideje ter " \
        "jih širiti po celotni druščini. Na te junake se lahko obrnemo, kadar želimo, da sporočilo ali nova" \
        " ideja doseže čim več članov ekipe. ONA nam omogoča, da prepoznamo ključne povezovalce, " \
        "povečamo sodelovanje in zagotovimo, da ekipa deluje učinkovito ter usklajeno, kar je odločilno, " \
        "če želimo skupaj rešiti svet."
    )

    if all(k in st.session_state for k in ["trust", "advice", "friends"]):
        matrices = {
            "Zaupanje": st.session_state["trust_matrix"],
            "Svetovanje": st.session_state["advice_matrix"],
            "Prijateljstvo": st.session_state["friends_matrix"],
            "Skupno": st.session_state["combined_matrix"]
        }

        # --- 1️⃣ Compute layouts and normalize coordinate span ---
        forceatlas2 = ForceAtlas2(
            outboundAttractionDistribution=False,
            linLogMode=False,
            adjustSizes=False,
            edgeWeightInfluence=2.0,
            jitterTolerance=1,
            barnesHutOptimize=True,
            barnesHutTheta=0.8,
            scalingRatio=0.5,
            strongGravityMode=False,
            gravity=15,
            verbose=False
        )

        layouts = {}
        all_degrees = []

        # First, compute layout and collect global max degree
        for name, matrix in matrices.items():
            G = nx.DiGraph()
            m = matrix.values
            nodes = matrix.index.tolist()
            for i, src in enumerate(nodes):
                for j, tgt in enumerate(nodes):
                    if m[i, j] > 0:
                        G.add_edge(src, tgt, weight=m[i, j])

            pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=1000)
            layouts[name] = pos

            wdeg = np.array([
                sum(d['weight'] for _, _, d in G.edges(node, data=True)) +
                sum(d['weight'] for _, _, d in G.in_edges(node, data=True))
                for node in G.nodes()
            ])
            all_degrees.extend(wdeg)

        # Global normalization factors
        global_min_x = min(min(p[0] for p in pos.values()) for pos in layouts.values())
        global_max_x = max(max(p[0] for p in pos.values()) for pos in layouts.values())
        global_min_y = min(min(p[1] for p in pos.values()) for pos in layouts.values())
        global_max_y = max(max(p[1] for p in pos.values()) for pos in layouts.values())
        global_max_degree = max(all_degrees)

        def normalize_positions(positions):
            def scale(v, minv, maxv):
                return (v - minv) / (maxv - minv) * 1000  # scale to 0–1000
            return {n: (scale(x, global_min_x, global_max_x), scale(y, global_min_y, global_max_y))
                    for n, (x, y) in positions.items()}

        layouts = {k: normalize_positions(v) for k, v in layouts.items()}

        # --- 2️⃣ Plot and Save SVG ---
        def plot_static_network(matrix, title, pos):
            m = matrix.values
            nodes = matrix.index.tolist()

            # Create igraph for metrics
            g = ig.Graph.Adjacency((m > 0).tolist(), mode="directed")
            g.vs["name"] = nodes
            g.es["weight"] = m[m.nonzero()]

            # --- Metrics ---
            indeg = np.array(g.degree(mode="in"))
            outdeg = np.array(g.degree(mode="out"))
            wdeg = np.array(g.strength(weights="weight", mode="all"))
            btw = np.array(g.betweenness(weights="weight", directed=True, normalized=True))
            close = np.array(g.closeness(weights="weight", normalized=True))
            pr = np.array(g.pagerank(weights="weight", directed=True))
            cluster_coef = np.array(g.transitivity_local_undirected(mode="zero"))
            coreness = np.array(g.coreness(mode="all"))

            # --- Node stats table ---
            node_stats = pd.DataFrame({
                "Node": nodes,
                "Weighted Degree": np.round(wdeg, 3),
                "In-Degree": indeg,
                "Out-Degree": outdeg,
                "Betweenness": np.round(btw, 3),
                "PageRank": np.round(pr, 3),
                "Closeness": np.round(close, 3),
                "Clustering": np.round(cluster_coef, 3),
                "Coreness": coreness,
            }).sort_values("Betweenness", ascending=False)

            # --- Global stats ---
            density = g.density()
            reciprocity = g.reciprocity()
            avg_deg = np.mean(wdeg)
            num_edges = g.ecount()
            avg_path = g.average_path_length(directed=True)
            global_stats = pd.DataFrame({
                "Density": [np.round(density, 3)],
                "Reciprocity %": [np.round(reciprocity * 100, 2)],
                "Edges": [num_edges],
                "Avg. Degree": [np.round(avg_deg, 2)],
                "Shortest Path": [np.round(avg_path, 3)]
            })

            # Build NetworkX for visualization
            G = nx.DiGraph()
            for i, src in enumerate(nodes):
                for j, tgt in enumerate(nodes):
                    if m[i, j] > 0:
                        G.add_edge(src, tgt, weight=m[i, j])

            # --- Color map ---
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["#f9dfb6", "#f9a22b"])
            norm = mcolors.Normalize(vmin=min(btw), vmax=max(btw))
            node_colors = {node: mcolors.to_hex(cmap(norm(b))) for node, b in zip(nodes, btw)}

            # --- Node sizes (normalized globally) ---
            node_sizes = {node: 150 + (w / global_max_degree) * 800 for node, w in zip(nodes, wdeg)}

            # --- Edge color mapping ---
            weight_colors = {1: "#e8e8e8", 2: "#797979", 3: "#3a3a3a"}

            # --- Plot setup ---
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_facecolor("white")
            ax.axis("off")

            # --- Draw edges by weight layer ---
            for weight in [1, 2, 3]:
                for (u, v, d) in G.edges(data=True):
                    if int(d["weight"]) == weight:
                        color = weight_colors.get(weight, "#808080")
                        x1, y1 = pos[u]
                        x2, y2 = pos[v]

                        # Compute unit vector from source to target
                        dx, dy = x2 - x1, y2 - y1
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist == 0:
                            continue
                        ux, uy = dx / dist, dy / dist

                        # --- Define push distances ---
                        if title.lower().startswith("skupno"):
                            start_offset = 30
                            end_offset = 30
                        else:
                            start_offset = 10
                            end_offset = 10

                        # --- Compute adjusted start/end positions ---
                        start = (x1 + ux * start_offset, y1 + uy * start_offset)
                        end = (x2 - ux * end_offset, y2 - uy * end_offset)

                        # --- Draw arrow ---
                        arrow = FancyArrowPatch(
                            start, end,
                            arrowstyle="-|>,head_length=1.6,head_width=0.7",
                            color=color,
                            mutation_scale=5,
                            lw=1.2,
                            alpha=1.0,
                            zorder=weight + 1
                        )
                        ax.add_patch(arrow)

            # --- Draw nodes ---
            for node in nodes:
                xy = pos.get(node)
                if xy is None:
                    continue  # skip missing nodes
                x, y = xy
                color = node_colors[node]
                size = node_sizes[node]
                if node.lower() == "tu ste vi":
                    ax.scatter(x, y, s=size, color=color, marker="*", zorder=10, edgecolors=color, linewidths=0.8)
                else:
                    ax.scatter(x, y, s=size, color=color, zorder=9, edgecolors=color, linewidths=0.5)

            # --- Draw labels ---
            for node in nodes:
                if node not in pos:
                    continue  # Skip if missing from layout
                x, y = pos[node]
                if node.lower() == "tu ste vi":
                    ax.text(x, y + 15, "TU STE VI", color="#003a6e", ha="center", va="center",
                            fontsize=8, fontweight="bold", zorder=12, style='italic')
                else:
                    ax.text(x, y + 8, node, color="black", ha="center", va="center", fontsize=9, zorder=11)

            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            
            # Build full path for the SVG (safe on Windows and cloud)
            safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", title)
            file_name = f"{safe_title}.svg"
            svg_path = os.path.join(BASE_DIR, file_name)

            # Save SVG
            plt.savefig(svg_path, format="svg", dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Load SVG
            with open(svg_path, "r", encoding="utf-8") as f:
                svg_text = f.read()

            # Display SVG in Streamlit
            st.components.v1.html(svg_text, height=600)

            return node_stats, global_stats

            # # Display SVG
            # st.image(svg_path, width='stretch')
            # return node_stats, global_stats


        # --- 3️⃣ Render All Networks ---
        for i, (name, matrix) in enumerate(matrices.items()):
            # Only expand the first one
            expanded_state = (i == 0)

            with st.expander(f" {name}", expanded=expanded_state):
                node_stats, global_stats = plot_static_network(matrix, name, layouts[name])

                with st.expander(" Centralne metrike omrežja"):
                    st.dataframe(node_stats, width='stretch')

                with st.expander(" Globalne metrike omrežja"):
                    st.dataframe(global_stats, width='stretch')

        with st.expander("Opis metrik"):
                    st.markdown("""
                    **Node** - Predstavlja posameznika v omrežju, ki ga analiziramo v kontekstu sodelovanja ali komunikacije.  
                    **Weighted Degree** - Pokaže, kako zelo je oseba vključena v omrežje, saj meri skupno količino interakcij ali zahtev, ki jih izmenjuje z drugimi.  
                    **In-Degree** - Kaže, koliko ljudi se obrača na posameznika, kar odraža njegovo zaupanje, strokovnost ali neformalni vpliv.  
                    **Out-Degree** - Pokaže, kako pogosto oseba sama išče informacije, podporo ali sodelovanje pri drugih.  
                    **Betweenness** - Ocenjuje, v kolikšni meri oseba deluje kot most med skupinami, kar je ključno za prenos informacij v omrežju.  
                    **PageRank** - Ocenjuje strateški vpliv osebe glede na to, kako pomembni so ljudje, s katerimi je povezana.  
                    **Closeness** - Pokaže, kako hitro lahko oseba doseže druge v omrežju, kar je pomembno za učinkovito koordinacijo in reševanje nalog.  
                    **Clustering** - Meri, kako tesno povezani so ljudje okoli posameznika, kar razkriva lokalno sodelovanje in zaupanje.  
                    **Coreness** - Ocenjuje, ali je oseba del jedra omrežja ali njegove periferije, kar odraža raven vključenosti v organizacijsko dinamiko.   
                    **Density** - Pokaže, kako povezano je celotno omrežje, kar odraža splošno kulturo sodelovanja ali prisotnost komunikacijskih ovir.  
                    **Reciprocity %** - Izraža, kolikšen delež odnosov je obojestranskih, kar kaže na stabilnost, zaupanje in enakovrednost odnosov.  
                    **Edges** - Predstavlja skupno število vseh medosebnih povezav, kar kaže na splošno aktivnost in povezanost v omrežju.  
                    **Avg. Degree** - Pokaže povprečno vključenost zaposlenih ter daje vpogled v tipično stopnjo sodelovanja v organizaciji.    
                    **Shortest Path** - Ocenjuje povprečno razdaljo med osebami, kar vpliva na hitrost in učinkovitost pretoka informacij v organizaciji.    
                                
                    """)

# --- TAB 5 --- Aplikacije
with tab5:
    st.write("")
    st.write("Primeri izzivov in priložnosti za izboljšave s pomočjo Analize Organizacijskih Omrežji (ONA)")
    st.write("")

    from streamlit.components.v1 import html

    html("""
        <div id="aplikacija">
        <style>
            /* Ensure the outer container is white */
            #aplikacija {
                background-color: white !important;
                padding: 0;
            }
            #aplikacija table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 0px;
                background-color: white !important;
            }
            #aplikacija th, #aplikacija td {
                padding: 4px 6px;
                text-align: left;
                border-bottom: 1px solid #ddd;
                font-family: "DejaVu Serif", serif;
                font-size: 16px;
                line-height: 1.3;
                background-color: white !important;
            }
            #aplikacija th {
                font-weight: 600;
            }

            #aplikacija td.col1, #aplikacija th.col1 {
                font-weight: bold;
            }
        </style>

        <table>
            <colgroup>
                <col style="width:15%">
                <col style="width:35%">
                <col style="width:50%">
            </colgroup>
            <thead>
                <tr>
                    <th>Interesna skupina</th>
                    <th>Izziv / Scenarij</th>
                    <th>Kako ONA pomaga</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="col1" rowspan="4">HR strokovnjaki <br>in kadrovske službe</td>
                    <td>&#x25AA; Neformalna komunikacija ne sledi organizacijski strukturi</td>
                    <td>&#x0226B; Z ONA odkrijete, kdo so dejanski povezovalci informacij in kje nastajajo informacijski “otoki”</td>
                </tr>
                <tr>
                    <td>&#x25AA; Neenakomerna obremenitev timov ali posameznikov</td>
                    <td>&#x0226B; Analiza razkrije preobremenjene posameznike in tiste, ki so izključeni iz ključnih tokov informacij</td>
                </tr>
                <tr>
                    <td>&#x25AA; Onboarding novih / Reboarding starih sodelavcev ni učinkovit</td>
                    <td>&#x0226B; Z ONA analizirate vključevanje novih članov v mrežo, kdo jim je dejanska podpora</td>
                </tr>
                <tr>
                    <td>&#x25AA; Kadri prehitro odhajajo</td>
                    <td>&#x0226B; Raziskava razkrije občutek izolacije ali izključenosti, ki vodi v fluktuacijo</td>
                </tr>

                <tr>
                    <td class="col1" rowspan="4">CEO, vodje podjetij, timski vodje</td>
                    <td>&#x25AA; Ekipa je rasla prehitro in komunikacija se zgošča v nekaj točkah</td>
                    <td>&#x0226B; Z ONA vidite, kje sistem postane preveč centraliziran ali kaotičen</td>
                </tr>
                <tr>
                    <td>&#x25AA; Več ekip sodeluje na skupnih projektih, a ne delijo znanja</td>
                    <td>&#x0226B; Analiza pokaže, kje so prekinjene povezave med ekipami in kje lahko izboljšate kroženje znanja</td>
                </tr>
                <tr>
                    <td>&#x25AA; Implementacija sprememb je počasna</td>
                    <td>&#x0226B; ONA identificira, kdo ima največ vpliva, da postane katalizator sprememb</td>
                </tr>
                <tr>
                    <td>&#x25AA; Vodstvo (CEO, direktorji) se nezavedno odmakne od preostale ekipe</td>
                    <td>&#x0226B; ONA razkrije, ali so vodje vključeni v vsakdanjo mrežo sodelovanja, ali pa informacije in povezave do njih redko potekajo. Tako lahko aktivno okrepijo prisotnost in vpliv v ključnih točkah sistema</td>
                </tr>

                <tr>
                    <td class="col1" rowspan="4">Coachi, svetovalci, psihologi</td>
                    <td>&#x25AA; Orgnizacija ne ve, zakaj intervencija ni učinkovita v celotnem timu</td>
                    <td>&#x0226B; Z ONA dobite objektivne podatke, ki pokažejo, kje je odpor ali nevidna dinamika</td>
                </tr>
                <tr>
                    <td>&#x25AA; Konflikti, ki se ne razrešijo</td>
                    <td>&#x0226B; Omrežna analiza razkrije, kdo je izključen, komu se člani izogibajo ali kje nastajajo tihe koalicije</td>
                </tr>
                <tr>
                    <td>&#x25AA; Vodje niso prepoznani kot oporni člen ekipe</td>
                    <td>&#x0226B; ONA razkrije razliko med formalno vlogo in dejanskim vplivom v mreži</td>
                </tr>
                <tr>
                    <td>&#x25AA; Ekipa je utrujena, motivacija pada</td>
                    <td>&#x0226B; S pomočjo večdimenzionalnih mrež (npr. pomoč, informacije, energija) razkrijemo, kako se prenaša pozitivna ali negativna energija</td>
                </tr>
            </tbody>
        </table>
        </div>
        """, height=900)

# --- TAB 6 --- Vizitka
with tab6:

    # Display the image
    st.image(str(ASSETS/"img1.png"), width='stretch')
    st.write("")
    st.markdown("<hr style='border: 1px dashed  #fac108; margin: 1rem 0;'>", unsafe_allow_html=True)
        # --- Here: single-line links with icons/images ---
    # Single line links with text only
    links_html = """
    <div style="display:flex; justify-content:center; align-items:center; gap:2rem; font-size:25px;">
        <a href="https://www.ictus.si/" target="_blank"> Spletna stran</a>
        <a href="mailto:info@ictus.si"> Email</a>
        <a href="https://www.linkedin.com/in/primozdolzan" target="_blank"> LinkedIn Primoz</a>
        <a href="https://si.linkedin.com/company/ictus-ona" target="_blank"> LinkedIn Ictus</a>
    </div>
    """
    st.markdown(links_html, unsafe_allow_html=True)
    st.write("")
    st.markdown("<hr style='border: 1px dashed  #fac108; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.write("")
    st.image(str(ASSETS/"img2.png"), width='stretch')

# --- TAB 7 --- Experimental interctivity
# --- TAB 7: Interaktivno omrežje ---
with tab7:

        st.markdown("""
        <style>

        /* === SCOPE EVERYTHING TO TAB 7 ONLY === */
        #tab7-scope * {
            font-size: 11px !important;
        }

        /* =====================================================
        SELECTBOX (closed state)
        ===================================================== */
        #tab7-scope div[data-testid="stSelectbox"] label p {
            font-size: 11px !important;
        }

        #tab7-scope div[data-testid="stSelectbox"] button span {
            font-size: 11px !important;
        }

        /* Box height + padding */
        #tab7-scope div[data-testid="stSelectbox"] button {
            min-height: 26px !important;
            padding: 2px 6px !important;
        }

        /* =====================================================
        MULTISELECT (closed state)
        ===================================================== */
        #tab7-scope div[data-testid="stMultiSelect"] label p {
            font-size: 11px !important;
        }

        #tab7-scope div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
            font-size: 10px !important;
            padding-top: 1px !important;
            padding-bottom: 1px !important;
        }

        #tab7-scope div[data-testid="stMultiSelect"] input {
            font-size: 11px !important;
        }

        /* Box height */
        #tab7-scope div[data-testid="stMultiSelect"] > div {
            min-height: 26px !important;
        }

        /* =====================================================
        DROPDOWN MENU — ACTUAL ITEMS (this is what you needed)
        ===================================================== */

        /* All dropdown item text */
        #tab7-scope div[role="listbox"] * {
            font-size: 8px !important;
            line-height: 1.1 !important;
        }

        /* Option label text */
        #tab7-scope div[role="option"],
        #tab7-scope div[role="option"] p,
        #tab7-scope div[role="option"] span {
            font-size: 11px !important;
            padding-top: 2px !important;
            padding-bottom: 2px !important;
        }

        /* =====================================================
        OPTIONAL: Remove big vertical spacing between controls
        ===================================================== */
        #tab7-scope .st-emotion-cache-1lcbmhc,
        #tab7-scope .st-emotion-cache-1kyxreq,
        #tab7-scope .st-emotion-cache-1y4p8pa {
            margin-top: 0px !important;
            margin-bottom: 0px !important;
        }

        </style>
        <div id="tab7-scope">
        """, unsafe_allow_html=True)


        # --- CONTROLS GRAPH (no columns) ---

        matrix_choice = st.selectbox(
            "Izberi matriko",
            ["Zaupanje", "Svetovanje", "Prijateljstvo"],
            index=0,
            key="matrix_choice_tab7"
        )

        size_metric = st.selectbox(
            "Velikost vozlišč (metrika)",
            ["Degree", "In-Degree", "Out-Degree", "Betweenness", "Closeness",
            "PageRank", "Clustering", "Coreness"],
            index=0,
            key="size_metric_tab7"
        )

        color_metric = st.selectbox(
            "Barva vozlišč (metrika)",
            ["Degree", "In-Degree", "Out-Degree", "Betweenness", "Closeness",
            "PageRank", "Clustering", "Coreness"],
            index=3,
            key="color_metric_tab7"
        )

        removed_nodes = st.multiselect(
            "Odstrani vozlišča",
            options=list(all_names),
            default=[],
            key="removed_nodes_tab7"
        )

        # --- SELECT MATRIX ---
        matrices = {
        "Zaupanje": st.session_state["trust_matrix"],
        "Svetovanje": st.session_state["advice_matrix"],
        "Prijateljstvo": st.session_state["friends_matrix"]
        }

        M = matrices[matrix_choice].copy()

        # Remove nodes if selected
        if removed_nodes:
            M = M.drop(index=removed_nodes, columns=removed_nodes)

        # BUILD GRAPH
        G = nx.from_pandas_adjacency(M, create_using=nx.DiGraph)

        # --- METRICS ---
        degree = dict(G.degree())
        indeg = dict(G.in_degree())
        outdeg = dict(G.out_degree())
        betweenness = nx.betweenness_centrality(G, normalized=True)
        closeness = nx.closeness_centrality(G)
        pagerank = nx.pagerank(G)
        clustering = nx.clustering(G.to_undirected())
        try:
            coreness = nx.core_number(G.to_undirected())
        except:
            coreness = {n: 0 for n in G.nodes()}

        metrics = {
            "Degree": degree,
            "In-Degree": indeg,
            "Out-Degree": outdeg,
            "Betweenness": betweenness,
            "Closeness": closeness,
            "PageRank": pagerank,
            "Clustering": clustering,
            "Coreness": coreness
        }

        # --- COLOR SCALE ---
        def color_for_value(val, vmin, vmax):
            if vmax == vmin:
                return "#f9dfb6"
            t = (val - vmin) / (vmax - vmin)
            from_color = (249, 223, 182)
            to_color   = (249, 162,  43)
            r = int(from_color[0] + t * (to_color[0] - from_color[0]))
            g = int(from_color[1] + t * (to_color[1] - from_color[1]))
            b = int(from_color[2] + t * (to_color[2] - from_color[2]))
            return f"rgb({r},{g},{b})"

        color_values = metrics[color_metric]
        min_val = min(color_values.values())
        max_val = max(color_values.values())

        # --- PYVIS NETWORK ---
        net = Network(height="700px", width="100%", bgcolor="white", directed=True)
        net.barnes_hut()  # physics enabled

        for node in G.nodes():
            val = metrics[size_metric][node]
            color = color_for_value(color_values[node], min_val, max_val)
            size = 50 + val * 30  # small natural scaling

            label = node

            # Special styling for TU STE VI
            shape = "star" if node == "TU STE VI" else "dot"

            net.add_node(
                node,
                label=label,
                shape=shape,
                color=color,
                size=size,
                font={'size': 120},
                title=label  # tooltip contains ONLY label
            )

        # Add directed edges
        for u, v in G.edges():
            net.add_edge(
                u, v,
                color="gray",
                width=3,   # thicker edges
                arrows={'to': {'enabled': True, 'scaleFactor': 8    }}  # ← larger arrows
    )

        # Generate HTML and display
        net.save_graph("interactive_network.html")
        HtmlFile = open("interactive_network.html", "r", encoding="utf-8")
        st.components.v1.html(HtmlFile.read(), height=750, scrolling=True)
