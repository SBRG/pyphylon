'''
Functions for analyzing genome synteny and comparing between strains in a phylon and between phylons
'''

import pandas as pd
import numpy as np
import networkx as nx

def get_phylon_strains(A_binarized, phylon):
    '''
    Get all strains associated with a given phylon
    '''
    return list(A_binarized.T[A_binarized.loc[phylon] == 1].index)

def get_phylon_genes(L_binarized, phylon):
    '''
    Get all genes associated with a given phylon
    '''

    return list(L_binarized[L_binarized[phylon] == 1].index)


def generate_strain_inputs(strain_vectors, metadata, target_strains):
    '''
    Function to create input for downstream plotting by taking strain vectors and generating ordered genes on each contig per strain of interest
    '''

    strain_inputs = {}
    
    metadata_check = metadata.set_index('genome_id')
    
    for strain in target_strains:
        df = strain_vectors[strain]
    
        strain_dict = {
            'strain_id': strain,
            'contigs': [],
            'metadata': {}
        }
    
        for contig in df.accession.unique():
            df_contig = df[df.accession == contig]
            genes = df_contig.gene.dropna().to_list()
    
            strain_dict['contigs'].append(genes)
        strain_dict['metadata'] = metadata_check.loc[strain].to_dict()
        strain_inputs[strain] = strain_dict

    return strain_inputs


    import networkx as nx


def build_pangenome_graph(
    strain_inputs,
    genes_to_keep=None,
    circular=True,
):
    """
    Build a gene adjacency graph from ordered contigs.

    Parameters
    ----------
    strain_inputs : dict
        Output from generate_strain_inputs().

    genes_to_keep : set, optional
        Only genes in this set are retained.

    circular : bool
        Whether contigs should wrap around.

    Returns
    -------
    networkx.Graph
    """

    G = nx.Graph()

    for strain, info in strain_inputs.items():

        for contig in info["contigs"]:

            if genes_to_keep is not None:
                contig = [g for g in contig if g in genes_to_keep]

            if len(contig) < 2:
                continue

            end = len(contig) if circular else len(contig) - 1

            for i in range(end):

                gene1 = contig[i]
                gene2 = contig[(i + 1) % len(contig)]

                if not G.has_node(gene1):
                    G.add_node(gene1, strains=set())

                if not G.has_node(gene2):
                    G.add_node(gene2, strains=set())

                G.nodes[gene1]["strains"].add(strain)
                G.nodes[gene2]["strains"].add(strain)

                if G.has_edge(gene1, gene2):
                    G.edges[gene1, gene2]["strains"].add(strain)
                else:
                    G.add_edge(gene1, gene2, strains={strain})

    return G


def annotate_graph(
    G,
    core_genes=None,
    accessory_genes=None,
    rare_genes=None,
):
    """
    Annotate graph nodes and edges with metadata.
    """

    core_genes = set(core_genes or [])
    accessory_genes = set(accessory_genes or [])
    rare_genes = set(rare_genes or [])

    for node in G.nodes:

        if node in rare_genes:
            category = "rare"
        elif node in accessory_genes:
            category = "accessory"
        else:
            category = "core"

        G.nodes[node]["num_strains"] = len(G.nodes[node]["strains"])
        G.nodes[node]["node_category"] = category

    for u, v in G.edges:
        G.edges[u, v]["num_strains"] = len(G.edges[u, v]["strains"])

    return G


def get_gene_subset(
    core_genes,
    accessory_genes=None,
    phylon_genes=None,
    mode="core_phylon",
):
    """
    Determine which genes are retained.

    mode:
        "all"
        "core"
        "core_phylon"
    """

    core_genes = set(core_genes)

    if mode == "all":
        return None

    if mode == "core":
        return core_genes

    if mode == "core_phylon":
        return core_genes | set(phylon_genes)

    raise ValueError(f"Unknown mode: {mode}")


def annotate_support(G):

    for node in G.nodes():

        G.nodes[node]["weight"] = len(
            G.nodes[node]["strains"]
        )

    for u,v in G.edges():

        G[u][v]["weight"] = len(
            G[u][v]["strains"]
        )

    return G

def normalize_graph_support(
    G,
    n_strains,
):
    """
    Normalize edge support by the number of strains
    in the phylon.
    """

    H = G.copy()

    for u, v in H.edges:

        H.edges[u, v]["weight"] = (
            H.edges[u, v]["num_strains"] / n_strains
        )

    return H


def complete_graph(
    H,
    penalty=1e-6,
):
    '''
    Add penalty edges to graph to ensure connectedness for performance
    '''
    G = H.copy()

    nodes = list(G.nodes)

    for i,u in enumerate(nodes):

        for v in nodes[i+1:]:

            if not G.has_edge(u,v):

                G.add_edge(
                    u,
                    v,
                    weight=penalty,
                )

    return G



def solve_max_weight_hamiltonian_cycle(
    H,
    time_limit=300000
):

    '''
    Solve the circular chromosome in the graph as a hamiltonian cycle using MILP
    '''
    from ortools.linear_solver import pywraplp
    import networkx as nx


    nodes = list(H.nodes)

    node_to_idx = {
        n:i for i,n in enumerate(nodes)
    }

    solver = pywraplp.Solver.CreateSolver(
        "CBC"
    )

    solver.SetTimeLimit(time_limit)


    x = {}

    for u,v in H.edges:

        i = node_to_idx[u]
        j = node_to_idx[v]

        x[i,j] = solver.BoolVar(
            f"x_{i}_{j}"
        )

        x[j,i] = x[i,j]


    # objective

    objective = solver.Objective()

    for u,v,data in H.edges(data=True):

        i=node_to_idx[u]
        j=node_to_idx[v]

        objective.SetCoefficient(
            x[i,j],
            data["weight"]
        )

    objective.SetMaximization()


    # degree constraints

    for node in nodes:

        i=node_to_idx[node]

        c = solver.Constraint(2,2)

        for j in range(len(nodes)):

            if (i,j) in x:

                c.SetCoefficient(
                    x[i,j],
                    1
                )


    # iterative subtour elimination

    while True:

        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            break


        selected = nx.Graph()

        selected.add_nodes_from(nodes)


        for (i,j),var in x.items():

            if i < j and var.solution_value() > 0.5:

                selected.add_edge(
                    nodes[i],
                    nodes[j]
                )


        components = list(
            nx.connected_components(
                selected
            )
        )


        if len(components) == 1:
            break


        print(
            "Adding subtour constraints:",
            [len(c) for c in components]
        )


        # eliminate each smaller cycle

        for comp in components:

            if len(comp)==len(nodes):
                continue

            constraint = solver.Constraint(
                -solver.infinity(),
                len(comp)-1
            )

            for u in comp:

                for v in comp:

                    if u >= v:
                        continue

                    i=node_to_idx[u]
                    j=node_to_idx[v]

                    if (i,j) in x:

                        constraint.SetCoefficient(
                            x[i,j],
                            1
                        )


    # extract final cycle

    cycle = nx.cycle_basis(selected)[0]

    cycle.append(
        cycle[0]
    )

    return cycle



def score_consensus(
    H,
    order,
):

    rows = []

    for u,v in zip(
        order[:-1],
        order[1:],
    ):

        rows.append(
            {
                "gene1":u,
                "gene2":v,
                "support":
                H[u][v]["weight"]
                if H.has_edge(u,v)
                else 0,
            }
        )

    return pd.DataFrame(rows)



def chromosome_similarity(
    consensus_order,
    strain_contig,
):

    position = {
        g:i
        for i,g in enumerate(consensus_order)
    }

    score = 0

    comparisons = 0

    for a,b in zip(
        strain_contig[:-1],
        strain_contig[1:],
    ):

        if (
            a not in position
            or
            b not in position
        ):
            continue

        comparisons += 1

        if abs(
            position[a]
            -
            position[b]
        ) == 1:

            score += 1

    if comparisons == 0:
        return np.nan

    return score/comparisons


'''
 PLOTTING FUNCTIONS
'''

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def rotate_order(order, anchor_gene):
    """
    Rotate a circular gene order so that anchor_gene is first.
    """

    if anchor_gene not in order:
        raise ValueError(f"{anchor_gene} not found.")

    i = order.index(anchor_gene)

    return order[i:] + order[:i]


def plot_strain_dotplot(
    consensus_order,
    contig,
    anchor_gene=None,
):

    # Rotate consensus
    if anchor_gene is not None:

        consensus_order = rotate_order(
            consensus_order,
            anchor_gene,
        )

    lookup = {
        gene: i
        for i, gene in enumerate(consensus_order)
    }

    # Rotate strain
    if (
        anchor_gene is not None
        and anchor_gene in contig
    ):

        contig = rotate_order(
            contig,
            anchor_gene,
        )

    x = []
    y = []

    for pos, gene in enumerate(contig):

        if gene in lookup:

            x.append(lookup[gene])
            y.append(pos)

    plt.figure(figsize=(8,8))

    plt.scatter(
        x,
        y,
        s=5,
    )

    plt.xlabel("Consensus position")
    plt.ylabel("Strain position")

    if anchor_gene is not None:
        plt.title(f"Aligned on {anchor_gene}")

    plt.tight_layout()
    plt.show()




def plot_backbone_comparison(
    backbone1,
    backbone2,
    name1="Backbone 1",
    name2="Backbone 2",
    anchor_gene="dnaA",
):

    ############################################################
    # Copy inputs
    ############################################################

    b1 = list(backbone1)
    b2 = list(backbone2)

    shared = set(b1) & set(b2)

    if len(shared) == 0:
        raise ValueError("No shared genes.")

    ############################################################
    # Orient backbone2
    ############################################################

    shared1 = [g for g in b1 if g in shared]
    shared2 = [g for g in b2 if g in shared]

    positions = [shared1.index(g) for g in shared2]

    if (
        len(positions) > 1
        and
        np.sum(np.diff(positions) < 0)
        >
        np.sum(np.diff(positions) > 0)
    ):
        b2.reverse()

    ############################################################
    # Rotate to anchor
    ############################################################

    if anchor_gene in b1:
        i = b1.index(anchor_gene)
        b1 = b1[i:] + b1[:i]

    if anchor_gene in b2:
        i = b2.index(anchor_gene)
        b2 = b2[i:] + b2[:i]

    ############################################################
    # Position maps
    ############################################################

    pos1 = {g: i for i, g in enumerate(b1)}
    pos2 = {g: i for i, g in enumerate(b2)}

    ############################################################
    # Neighbor maps (circular chromosomes)
    ############################################################

    next2 = {
        b2[i]: b2[(i + 1) % len(b2)]
        for i in range(len(b2))
    }

    prev2 = {
        b2[i]: b2[(i - 1) % len(b2)]
        for i in range(len(b2))
    }

    ############################################################
    # Shared order
    ############################################################

    shared_order = [
        g
        for g in b1
        if g in shared
    ]

    ############################################################
    # Classify conserved adjacencies
    ############################################################

    adjacency_type = {}

    for g1, g2 in zip(shared_order[:-1], shared_order[1:]):

        if next2.get(g1) == g2:

            adjacency_type[(g1, g2)] = "conserved"

        elif prev2.get(g1) == g2:

            adjacency_type[(g1, g2)] = "reversed"

        else:

            adjacency_type[(g1, g2)] = "broken"

    ############################################################
    # Unique genes
    ############################################################

    unique1 = [
        g
        for g in b1
        if g not in shared
    ]

    unique2 = [
        g
        for g in b2
        if g not in shared
    ]

    ############################################################
    # Plot
    ############################################################

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.hlines(
        2,
        0,
        len(b1) - 1,
        color="black",
        lw=4,
        alpha=0.20,
    )

    ax.hlines(
        1,
        0,
        len(b2) - 1,
        color="steelblue",
        lw=4,
        alpha=0.20,
    )

    ############################################################
    # Draw correspondence lines
    ############################################################

    previous = None

    for gene in shared_order:

        color = "forestgreen"
        alpha = 0.18
        lw = 0.8

        if previous is not None:

            relation = adjacency_type[(previous, gene)]

            if relation == "reversed":

                color = "darkorange"
                alpha = 0.8
                lw = 2.3

            elif relation == "broken":

                color = "purple"
                alpha = 0.8
                lw = 2.3

        ax.plot(
            [pos1[gene], pos2[gene]],
            [2, 1],
            color=color,
            alpha=alpha,
            lw=lw,
            zorder=2,
        )

        previous = gene

    ############################################################
    # Backbone nodes
    ############################################################

    ax.scatter(
        range(len(b1)),
        [2] * len(b1),
        color="black",
        s=10,
        zorder=3,
    )

    ax.scatter(
        range(len(b2)),
        [1] * len(b2),
        color="steelblue",
        s=10,
        zorder=3,
    )

    ############################################################
    # Genes unique to backbone1
    ############################################################

    if unique1:

        ax.scatter(
            [pos1[g] for g in unique1],
            [2] * len(unique1),
            marker="x",
            color="cyan",
            s=40,
            zorder=4,
        )

    ############################################################
    # Genes unique to backbone2
    ############################################################

    if unique2:

        ax.scatter(
            [pos2[g] for g in unique2],
            [1] * len(unique2),
            marker="o",
            color="magenta",
            s=20,
            zorder=4,
        )

    ############################################################
    # Anchor
    ############################################################

    ax.scatter(
        0,
        2,
        marker="*",
        s=160,
        color="gold",
        edgecolor="black",
        zorder=5,
    )

    ax.scatter(
        0,
        1,
        marker="*",
        s=160,
        color="gold",
        edgecolor="black",
        zorder=5,
    )

    ############################################################
    # Cosmetics
    ############################################################

    ax.set_xlim(
        -2,
        max(len(b1), len(b2)) + 2,
    )

    ax.set_ylim(
        0.5,
        2.5,
    )

    ax.set_xlabel("Aligned genomic position")

    ax.set_yticks([1, 2])

    ax.set_yticklabels(
        [name2, name1],
        fontsize=12,
    )

    ############################################################
    # Legend
    ############################################################

    legend = [

        Line2D(
            [0], [0],
            color="forestgreen",
            lw=2,
            label="Conserved adjacency",
        ),

        Line2D(
            [0], [0],
            color="darkorange",
            lw=2,
            label="Reversed adjacency",
        ),

        Line2D(
            [0], [0],
            color="purple",
            lw=2,
            label="Broken adjacency",
        ),

        Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            color="cyan",
            markersize=7,
            label=f"Unique to {name1}",
        ),

        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color="magenta",
            markersize=6,
            label=f"Unique to {name2}",
        ),

        Line2D(
            [0], [0],
            marker="*",
            linestyle="None",
            color="gold",
            markeredgecolor="black",
            markersize=12,
            label=f"Anchor ({anchor_gene})",
        ),
    ]

    ax.legend(
        handles=legend,
        loc="upper right",
        frameon=True,
    )

    plt.tight_layout()

    return fig, ax


def plot_consensus_synteny(
    consensus_order,
    strain_inputs,
    figsize=(20,10),
    max_insert_display=10,
):
    """
    Plot every strain projected onto the consensus chromosome.

    Optimization:
        Draw all points for each strain with only two scatter calls.
    """

    consensus = set(consensus_order)

    x_lookup = {
        g: i
        for i, g in enumerate(consensus_order)
    }

    fig, ax = plt.subplots(figsize=figsize)

    strains = list(strain_inputs.keys())

    for row, strain in enumerate(strains):

        y = len(strains) - row

        ##################################################
        # Build observed set
        ##################################################

        observed = set()

        for contig in strain_inputs[strain]["contigs"]:
            observed.update(contig)

        ##################################################
        # Collect points
        ##################################################

        x_present = []
        y_present = []

        x_missing = []
        y_missing = []

        for gene in consensus_order:

            if gene in observed:

                x_present.append(x_lookup[gene])
                y_present.append(y)

            else:

                x_missing.append(x_lookup[gene])
                y_missing.append(y)

        ##################################################
        # Draw all points at once
        ##################################################

        ax.scatter(
            x_present,
            y_present,
            color="black",
            s=6,
            zorder=3,
        )

        ax.scatter(
            x_missing,
            y_missing,
            color="red",
            marker="x",
            s=18,
            zorder=3,
        )

        ##################################################
        # Draw edges
        ##################################################

        for contig in strain_inputs[strain]["contigs"]:

            positions = {
                gene: i
                for i, gene in enumerate(contig)
            }

            conserved = [
                g
                for g in contig
                if g in consensus
            ]

            for g1, g2 in zip(
                conserved[:-1],
                conserved[1:]
            ):

                x1 = x_lookup[g1]
                x2 = x_lookup[g2]

                i1 = positions[g1]
                i2 = positions[g2]

                inserted = [
                    g
                    for g in contig[i1+1:i2]
                    if g not in consensus
                ]

                if abs(x2 - x1) == 1:

                    if len(inserted) == 0:

                        ax.plot(
                            [x1, x2],
                            [y, y],
                            color="black",
                            lw=1,
                            alpha=0.7,
                        )

                    else:

                        ax.plot(
                            [x1, x2],
                            [y, y],
                            color="orange",
                            lw=min(
                                1 + len(inserted),
                                max_insert_display,
                            ),
                            alpha=0.8,
                        )

                else:

                    ax.plot(
                        [x1, x2],
                        [y, y],
                        color="dodgerblue",
                        lw=0.8,
                        alpha=0.4,
                    )

    ax.set_xlim(-1, len(consensus_order))
    ax.set_ylim(0, len(strains) + 1)

    ax.set_xlabel("Consensus chromosome")
    ax.set_ylabel("Strains")

    ax.set_yticks(np.arange(1, len(strains) + 1))
    ax.set_yticklabels(
        strains[::-1],
        fontsize=6,
    )

    plt.tight_layout()

    legend_elements = [

        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Conserved gene present",
            markerfacecolor="black",
            markersize=5,
        ),
    
        Line2D(
            [0], [0],
            marker="x",
            color="red",
            label="Conserved gene missing",
            linestyle="None",
            markersize=6,
        ),
    
        Line2D(
            [0], [0],
            color="black",
            lw=2,
            label="Conserved adjacency",
        ),
    
        Line2D(
            [0], [0],
            color="orange",
            lw=3,
            label="Accessory insertion",
        ),
    
        Line2D(
            [0], [0],
            color="dodgerblue",
            lw=2,
            label="Rearranged adjacency",
        ),
    
    ]
    
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
    )

    return fig, ax



def plot_accessory_insertions(
    consensus_order,
    strain_inputs,
):

    consensus = set(consensus_order)

    insertion_counts = np.zeros(
        len(consensus_order)-1
    )

    lookup = {
        g:i
        for i,g in enumerate(consensus_order)
    }

    for strain in strain_inputs.values():

        for contig in strain["contigs"]:

            conserved = [
                g
                for g in contig
                if g in consensus
            ]

            for a,b in zip(
                conserved[:-1],
                conserved[1:]
            ):

                ia = lookup[a]
                ib = lookup[b]

                if abs(ib-ia) != 1:
                    continue

                i1 = contig.index(a)
                i2 = contig.index(b)

                insertion = contig[
                    i1+1:i2
                ]

                insertion = [
                    g
                    for g in insertion
                    if g not in consensus
                ]

                insertion_counts[
                    min(ia,ib)
                ] += len(insertion)

    plt.figure(figsize=(16,4))

    plt.plot(
        insertion_counts
    )

    plt.xlabel(
        "Consensus interval"
    )

    plt.ylabel(
        "Inserted Genes"
    )

    plt.tight_layout()

    plt.show()




def plot_presence_heatmap(
    consensus_order,
    strain_inputs,
):

    genes = consensus_order

    matrix = np.zeros(
        (
            len(strain_inputs),
            len(genes)
        ),
        dtype=int,
    )

    strains = list(strain_inputs.keys())

    gene_to_col = {
        g:i
        for i,g in enumerate(genes)
    }

    for r,strain in enumerate(strains):

        present = set()

        for contig in strain_inputs[strain]["contigs"]:
            present.update(contig)

        for gene in present:

            if gene in gene_to_col:

                matrix[
                    r,
                    gene_to_col[gene]
                ] = 1

    plt.figure(
        figsize=(20,8)
    )

    plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap = 'Blues',
    )

    plt.xlabel("Consensus position")
    plt.ylabel("Strain")

    plt.yticks(
        range(len(strains)),
        strains,
        fontsize=6,
    )

    plt.colorbar(
        label="Present"
    )

    plt.tight_layout()
    plt.show()


def plot_edge_support(edge_scores):

    plt.figure(figsize=(16,4))

    plt.plot(
        edge_scores["support"].values,
        lw=1
    )

    plt.xlabel("Consensus position")
    plt.ylabel("Normalized edge support")
    plt.ylim(0,1.05)

    plt.tight_layout()
    plt.show()
