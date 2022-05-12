from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

def plot_barplot_of_tags(tags_list, nb_of_tags, xlabel="Nombre d'occurences", ylabel="", figsave=None, figsize=(10,30)):
    """
    Description: plot barplot of tags count (descending order) from a list of tags
    
    Args:
        - tags_list (lsit): list of tags
        - nb_of_tags (int) : number of tags to plot in barplot (default=50)
        - xlabel, ylabel (str): labels of the barplot
        - figsize (list) : figure size (default : (10, 30))
        
    Output :
        - Barplot of nb_of_tags most important tags
    """
    
    tag_count = Counter(tags_list)
    tag_count_sort = dict(tag_count.most_common(nb_of_tags))
    
    plt.figure(figsize=figsize)
    sns.barplot(x=list(tag_count_sort.values()), y=list(tag_count_sort.keys()), orient='h')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if figsave:
        plt.savefig(figsave, bbox_inches='tight')
    plt.show()
    

def plot_wordcloud(tags_list, figsave=None):
    """
    Description: plot wordcloud of most important tags from a list of tags
    
    Args:
        - tags_lists (series): list of tag lists
        - figsave(str) : name of the figure if want to save it 
        
    Output :
        - Wordcloud of tags, based on tag counts
    """
    
    word_cloud_dict=Counter(tags_list)
    wordcloud = WordCloud(background_color="white",width = 1000, height = 500).generate_from_frequencies(word_cloud_dict)

    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    
    if figsave:
        plt.savefig(figsave)
    plt.show()
    plt.close()
    
    
def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    

def plot_ecoscore_gauge(value, product, ref_value=None, brand=None, figsave=None):    
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = value,
        mode = "gauge+number+delta",
        title = {'text': str(f"{product} ({brand})")},
        delta = {'reference': ref_value},
        gauge = {'axis': {'range': [0, 11]},
                'bar': {'color': "grey"},
                 'steps' : [
                     {'range': [0, 1.8], 'color': "red"}, 
                     {'range': [1.8, 3.5], 'color': "orange"},
                     {'range': [3.5, 5.3], 'color': "yellow"},
                     {'range': [5.3, 7.2], 'color': "lightgreen"}, 
                     {'range': [7.2, 11], 'color': "darkgreen"}]
                }))
    
    if figsave:
        plt.savefig(figsave)
        
    fig.show()
    

    
def plot_radar_scores(df, index, figsave=None):    
    # Prepare data
    scores_df = df.loc[index,['health_score', 'ethi_score', 'sustainable_score', 'carbon_score', 'zero_waste_score']]

    # Plot radar_plot
    fig = px.line_polar(scores_df, r=scores_df.values, theta=scores_df.index, line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[-0.1, 2.5]
        )),
      showlegend=True
    )
    if figsave:
        plt.savefig(figsave)
        
    fig.show()
    

def plot__multiple_radar_scores(df, variables, figsave=None):
    fig = go.Figure()
    
    for i in range(df.shape[0]):
        subset = df.iloc[i]
        values = subset[:-1].values
        name = subset.values[-1]
        fig.add_trace(go.Scatterpolar(
          r=values,
          theta=variables,
          fill='toself',
          name=name
        ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[-0.1, 2.5]
        )),
      showlegend=True
    )

    if figsave:
        plt.savefig(figsave)
        
    fig.show()