import logging
import re
import urllib
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gzip
import pickle
from tqdm.notebook import tqdm, trange
import multiprocessing
from IPython.display import display, HTML
import itertools

import plotly.graph_objects as go
from plotly.subplots import make_subplots


anchor_gene_color = '#6495ED'
background_gene_color = '#F0F0F0'
phylon_gene_location = '#FFC96F'
Normal_Order_color = '#C6EBC5'
Inversion_Color = '#FA7070'
unique_gene_color = '#FF0000'


def plot_gene_sets_with_common_subset_proportional(gene_lists, common_subset, strain_names, X, Y):
    # Number of plots
    num_plots = len(gene_lists)

    # Create a subplot figure, with rows equal to the number of gene lists
    fig = make_subplots(rows=num_plots, cols=1, subplot_titles=strain_names)

    # Maximum number of genes in the lists to define x-axis range
    max_length = max(len(gene_list) for gene_list in gene_lists)  # Used for red lines

    # Add a bar for legend for each color once
    fig.add_trace(go.Bar(
        x=[0], y=[0], marker_color='black', name='Phylon', showlegend=True
    ))
    fig.add_trace(go.Bar(
        x=[0], y=[0], marker_color='lightgrey', name='Other Genes', showlegend=True
    ))
    
    # Plot each gene list in its own subplot
    for index, gene_list in enumerate(gene_lists, start=1):
        # Normalize x_values to use full width of subplot by distributing genes evenly
        x_values = [i * (max_length - 1) / (max(len(gene_list) - 1, 1)) for i in range(len(gene_list))]
        # Colors based on membership in the common subset
        colors = ['black' if gene in common_subset else 'lightgrey' for gene in gene_list]

        # Add bars with calculated x positions
        fig.add_trace(go.Bar(
            x=x_values,
            y=[1] * len(gene_list),
            marker_color=colors,
            width=0.9,  # Adjust the width to fit within subplot without touching red lines
            showlegend=False
        ), row=index, col=1)

    # Add continuous red boundary lines across all subplots
    # Ensure lines are outside the range of x-values used for gene bars
    fig.add_shape(type="line",
                  x0=-0.5, y0=0, x1=-0.5, y1=num_plots-(len(gene_lists)-1),  # Start boundary line
                  line=dict(color="red", width=6),
                  xref="x", yref="paper")

    
    fig.add_shape(type="line",
                  x0=max_length - 0.5, y0=0, x1=max_length - 0.5, y1=num_plots-(len(gene_lists)-1),  # End boundary line
                  line=dict(color="red", width=6),
                  xref="x", yref="paper")
        
    fig.add_annotation(x=0, y=num_plots-(len(gene_lists)-1)+0.05, xref="paper", yref="paper", font=dict(
            size=16,
            color="red"
            ), text=f"Anchor Gene {X}", showarrow=False)
    
    fig.add_annotation(x=1, y=num_plots-(len(gene_lists)-1)+0.05, xref="paper", yref="paper", font=dict(
            size=16,
            color="red"
            ), text=f"Anchor Gene {Y}", showarrow=False)
    
    # Update layout for better view
    fig.update_layout(
        title='',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=300 * num_plots,  # Adjust height based on number of plots
        bargap=0  # Remove any gap between bars
    )

    fig.show()


import plotly.graph_objects as go

def plot_circular_genome_combined_with_eggnog(list1, list2, title, strain, df_eggnog, show_legend=False):
    # Ensure the gene list starts with the number 1
    if 1 in list1:
        while list1[0] != 1:
            list1 = list1[-1:] + list1[:-1]

    # Calculate the number of genes
    num_genes = len(list1)

    # Define the angles for each gene, ensuring gene '1' is at the top center (90 degrees)
    angles = [(-i * 360 / num_genes + 90) % 360 for i in range(num_genes)]

    # Define the radial range for the bars
    inner_radius = 0.8
    outer_radius = 1.0
    
    outer_gap = 0.1  # Define the gap size between the inner and outer rings

    # Adjust base radius to create a gap
    outer_base_radius = outer_radius + outer_gap
    
    # Calculate the height of each bar
    bar_height = outer_radius - inner_radius

    # Create lists for the radii of each type of gene
    r_black = [bar_height] * num_genes
    r_blue = [bar_height if gene in list2 and not isinstance(gene, int) else 0 for gene in list1]
    r_red = [bar_height if isinstance(gene, int) else 0 for gene in list1]

    # Create hover text for each gene
    hover_text = []
    for gene in list1:
        if gene in df_eggnog.index:
            gene_info = df_eggnog.loc[gene]
            hover_text.append(
                f"{gene}<br>COG_category: {gene_info['COG_category']}<br>Preferred_name: {gene_info['Preferred_name']}<br>PFAMs: {gene_info['PFAMs']}<br>BiGG_Reaction: {gene_info['BiGG_Reaction']}"
            )
        else:
            hover_text.append(f"{gene}<br>COG_category: N/A<br>Preferred_name: N/A<br>PFAMs: N/A<br>BiGG_Reaction: N/A")

    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Barpolar(
        r=r_black,
        theta=angles,
        width=[360 / num_genes] * num_genes,
        base=[inner_radius] * num_genes,
        marker_color=background_gene_color,
        marker_line_color=background_gene_color,
        opacity=0.7,
        name='Genes',
        text=hover_text,
        hoverinfo='text',
        showlegend=show_legend
    ))

    # Add blue bars for genes in list2
    fig.add_trace(go.Barpolar(
        r=r_blue,
        theta=angles,
        width=[360 / num_genes] * num_genes,
        base=[inner_radius if gene in list2 and not isinstance(gene, int) else 0 for gene in list1],
        marker_color=phylon_gene_location,
        marker_line_color=phylon_gene_location,
        opacity=0.7,
        name=f'{title} Phylon genes',
        text=hover_text,
        hoverinfo='text',
        showlegend=show_legend
    ))

    # Add red bars for anchor genes (integers)
    fig.add_trace(go.Barpolar(
        r=r_red,
        theta=angles,
        width=[360 / num_genes] * num_genes,
        base=[inner_radius if isinstance(gene, int) else 0 for gene in list1],
        marker_color=anchor_gene_color,
        marker_line_color=anchor_gene_color,
        opacity=0.7,
        name='Anchor genes',
        text=hover_text,
        hoverinfo='text',
        showlegend=show_legend
    ))

    # Filter out non-integer genes for the outer ring
    int_genes = [gene for gene in list1 if isinstance(gene, int)]
    num_int_genes = len(int_genes)

    if num_int_genes > 0:
        # Calculate segment angles and widths based on positions of integer genes
        segment_base_angles = []
        segment_widths = []
        outer_colors = []
        outer_hover_text = []

        for i in range(num_int_genes):
            current_gene = int_genes[i]
            next_gene = int_genes[(i + 1) % num_int_genes]

            current_gene_index = list1.index(current_gene)
            next_gene_index = list1.index(next_gene)

            if next_gene_index < current_gene_index:
                next_gene_index += num_genes  # handle wrapping around the list

            # Calculate the angular width of the segment
            segment_width = (next_gene_index - current_gene_index) * (360 / num_genes)
            segment_widths.append(segment_width)

            # The base angle is the angle of the current gene
            base_angle = angles[current_gene_index] - segment_width / 2
            segment_base_angles.append(base_angle)

            # Determine the color of the segment
            if (current_gene < next_gene) or (i == num_int_genes - 1 and current_gene > next_gene):
                outer_colors.append(Normal_Order_color)  # Increasing order
            else:
                outer_colors.append(Inversion_Color)  # Decreasing order

            outer_hover_text.append(f'{current_gene}-{next_gene}')

        fig.add_trace(go.Barpolar(
            r=[bar_height] * num_int_genes,
            theta=segment_base_angles,
            width=segment_widths,
            base=[outer_base_radius] * num_int_genes,
            marker_color=outer_colors,
            marker_line_color=outer_colors,
            opacity=0.7,
            name='Outer Ring',
            text=outer_hover_text,
            hoverinfo='text',
            showlegend=show_legend
        ))

    # Update layout to add a white circle in the center
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, outer_base_radius + bar_height]),
            angularaxis=dict(visible=False),
            bgcolor="white"
        ),
        showlegend=show_legend,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig

def plot_combined_circular_genomes_with_variaton(figures, titles, phylon, figsize=(1500, 500), save_path=None, dpi=300, show = False):
    # Create a subplot layout with 1 row and len(figures) columns
    fig = make_subplots(rows=1, cols=len(figures), subplot_titles=titles, specs=[[{'type': 'polar'}]*len(figures)])

    for i, figure in enumerate(figures):
        for trace in figure['data']:
            trace.showlegend = False  # Hide legend for individual traces
            fig.add_trace(trace, row=1, col=i+1)

    # Add a single legend manually
    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None], mode='markers', marker=dict(color=background_gene_color, symbol='square', size=12), name='Genes', showlegend=True
    ))
    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None], mode='markers', marker=dict(color=phylon_gene_location, symbol='square', size=12), name=f'{phylon} Phylon Genes', showlegend=True
    ))
    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None], mode='markers', marker=dict(color=anchor_gene_color, symbol='square', size=12), name='Anchor Genes', showlegend=True
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None], mode='markers', marker=dict(color='black', symbol='square', size=0), name='', showlegend=True
    ))


    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None], mode='markers', marker=dict(color=Normal_Order_color, symbol='square', size=12), name='No variation', showlegend=True
    ))

    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None], mode='markers', marker=dict(color=Inversion_Color, symbol='square', size=12), name='Inversion', showlegend=True
    ))
    # Update the layout
    fig.update_layout(
        title = f'{phylon} Phylon Location',
        showlegend=True,
        legend=dict(x=1.1, y=0.65),  # Position the legend to the right of the last subplot
        width=figsize[0],
        height=figsize[1],
        paper_bgcolor='white',
        font=dict(
            size=14  # Adjust the font size as needed
        ),
        plot_bgcolor='white'
    )

    # Update each subplot individually to hide the angular axis and make the center white
    for i in range(1, len(figures) + 1):
        fig.update_polars(
            radialaxis=dict(visible=False),
            angularaxis=dict(visible=False),
            bgcolor="white"
        )
        
    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()


def plot_circular_genome_combined_with_eggnog_and_unique_genes(list1, list2, list3, title, strain, df_eggnog, show_legend=False):
    # Ensure the gene list starts with the number 1
    if 1 in list1:
        while list1[0] != 1:
            list1 = list1[-1:] + list1[:-1]

    # Calculate the number of genes
    num_genes = len(list1)

    # Define the angles for each gene, ensuring gene '1' is at the top center (90 degrees)
    angles = [(-i * 360 / num_genes + 90) % 360 for i in range(num_genes)]

    # Define the radial range for the bars
    inner_radius = 0.8
    outer_radius = 1.0
    
    outer_gap = 0.1  # Define the gap size between the inner and outer rings

    # Adjust base radius to create a gap
    outer_base_radius = outer_radius + outer_gap
    
    # Calculate the height of each bar
    bar_height = outer_radius - inner_radius

    # Create lists for the radii of each type of gene
    r_black = [bar_height] * num_genes
    r_blue = [bar_height if gene in list2 and not isinstance(gene, int) else 0 for gene in list1]
    r_red = [bar_height if isinstance(gene, int) else 0 for gene in list1]
    r_unique = [bar_height if gene in list3 and not isinstance(gene, int) else 0 for gene in list1]

    # Create hover text for each gene
    hover_text = []
    for gene in list1:
        if gene in df_eggnog.index:
            gene_info = df_eggnog.loc[gene]
            hover_text.append(
                f"{gene}<br>COG_category: {gene_info['COG_category']}<br>Preferred_name: {gene_info['Preferred_name']}<br>PFAMs: {gene_info['PFAMs']}<br>BiGG_Reaction: {gene_info['BiGG_Reaction']}"
            )
        else:
            hover_text.append(f"{gene}<br>COG_category: N/A<br>Preferred_name: N/A<br>PFAMs: N/A<br>BiGG_Reaction: N/A")

    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Barpolar(
        r=r_black,
        theta=angles,
        width=[360 / num_genes] * num_genes,
        base=[inner_radius] * num_genes,
        marker_color=background_gene_color,
        marker_line_color=background_gene_color,
        opacity=0.7,
        name='Genes',
        text=hover_text,
        hoverinfo='text',
        showlegend=show_legend
    ))

    # Add blue bars for genes in list2
    fig.add_trace(go.Barpolar(
        r=r_blue,
        theta=angles,
        width=[360 / num_genes] * num_genes,
        base=[inner_radius if gene in list2 and not isinstance(gene, int) else 0 for gene in list1],
        marker_color=phylon_gene_location,
        marker_line_color=phylon_gene_location,
        opacity=0.7,
        name=f'{title} Phylon genes',
        text=hover_text,
        hoverinfo='text',
        showlegend=show_legend
    ))

    # Add red bars for anchor genes (integers)
    fig.add_trace(go.Barpolar(
        r=r_red,
        theta=angles,
        width=[360 / num_genes] * num_genes,
        base=[inner_radius if isinstance(gene, int) else 0 for gene in list1],
        marker_color=anchor_gene_color,
        marker_line_color=anchor_gene_color,
        opacity=0.7,
        name='Anchor genes',
        text=hover_text,
        hoverinfo='text',
        showlegend=show_legend
    ))

    # Add unique bars for genes in list3
    fig.add_trace(go.Barpolar(
        r=r_unique,
        theta=angles,
        width=[360 / num_genes] * num_genes,
        base=[inner_radius if gene in list3 and not isinstance(gene, int) else 0 for gene in list1],
        marker_color=unique_gene_color,
        marker_line_color=unique_gene_color,
        opacity=0.7,
        name=f'Unique {title} Phylon genes',
        text=hover_text,
        hoverinfo='text',
        showlegend=show_legend
    ))

    # Filter out non-integer genes for the outer ring
    int_genes = [gene for gene in list1 if isinstance(gene, int)]
    num_int_genes = len(int_genes)

    if num_int_genes > 0:
        # Calculate segment angles and widths based on positions of integer genes
        segment_base_angles = []
        segment_widths = []
        outer_colors = []
        outer_hover_text = []

        for i in range(num_int_genes):
            current_gene = int_genes[i]
            next_gene = int_genes[(i + 1) % num_int_genes]

            current_gene_index = list1.index(current_gene)
            next_gene_index = list1.index(next_gene)

            if next_gene_index < current_gene_index:
                next_gene_index += num_genes  # handle wrapping around the list

            # Calculate the angular width of the segment
            segment_width = (next_gene_index - current_gene_index) * (360 / num_genes)
            segment_widths.append(segment_width)

            # The base angle is the angle of the current gene
            base_angle = angles[current_gene_index] - segment_width / 2
            segment_base_angles.append(base_angle)

            # Determine the color of the segment
            if (current_gene < next_gene) or (i == num_int_genes - 1 and current_gene > next_gene):
                outer_colors.append(Normal_Order_color)  # Increasing order
            else:
                outer_colors.append(Inversion_Color)  # Decreasing order

            outer_hover_text.append(f'{current_gene}-{next_gene}')

        fig.add_trace(go.Barpolar(
            r=[bar_height] * num_int_genes,
            theta=segment_base_angles,
            width=segment_widths,
            base=[outer_base_radius] * num_int_genes,
            marker_color=outer_colors,
            marker_line_color=outer_colors,
            opacity=0.7,
            name='Outer Ring',
            text=outer_hover_text,
            hoverinfo='text',
            showlegend=show_legend
        ))

    # Update layout to add a white circle in the center
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, outer_base_radius + bar_height]),
            angularaxis=dict(visible=False),
            bgcolor="white"
        ),
        showlegend=show_legend,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig

def plot_combined_circular_genomes_with_variaton_and_unique_genes(figures, titles, phylon, figsize=(1500, 500), save_path=None, dpi=300, show = False):
    # Create a subplot layout with 1 row and len(figures) columns
    fig = make_subplots(rows=1, cols=len(figures), subplot_titles=titles, specs=[[{'type': 'polar'}]*len(figures)])

    for i, figure in enumerate(figures):
        for trace in figure['data']:
            trace.showlegend = False  # Hide legend for individual traces
            fig.add_trace(trace, row=1, col=i+1)

    # Add a single legend manually

    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None], mode='markers', marker=dict(color=phylon_gene_location, symbol='square', size=12), name=f'{phylon} Phylon Genes', showlegend=True
    ))

    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None], mode='markers', marker=dict(color=unique_gene_color, symbol='square', size=12), name=f'Unique {phylon} Phylon Genes', showlegend=True
    ))
    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None], mode='markers', marker=dict(color=anchor_gene_color, symbol='square', size=12), name='Anchor Genes', showlegend=True
    ))
    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None], mode='markers', marker=dict(color=background_gene_color, symbol='square', size=12), name='Genes', showlegend=True
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None], mode='markers', marker=dict(color='black', symbol='square', size=0), name='', showlegend=True
    ))


    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None], mode='markers', marker=dict(color=Normal_Order_color, symbol='square', size=12), name='No variation', showlegend=True
    ))

    fig.add_trace(go.Scatterpolar(
        r=[None], theta=[None], mode='markers', marker=dict(color=Inversion_Color, symbol='square', size=12), name='Inversion', showlegend=True
    ))
    # Update the layout
    fig.update_layout(
        title = f'{phylon} Phylon Location',
        showlegend=True,
        legend=dict(x=1.1, y=0.65),  # Position the legend to the right of the last subplot
        width=figsize[0],
        height=figsize[1],
        paper_bgcolor='white',
        font=dict(
            size=14  # Adjust the font size as needed
        ),
        plot_bgcolor='white'
    )

    # Update each subplot individually to hide the angular axis and make the center white
    for i in range(1, len(figures) + 1):
        fig.update_polars(
            radialaxis=dict(visible=False),
            angularaxis=dict(visible=False),
            bgcolor="white"
        )
        
    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
