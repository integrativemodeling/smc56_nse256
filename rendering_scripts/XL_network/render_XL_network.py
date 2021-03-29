""" 
This script is used to generate scaffolds for Figs 2a, 2b, 2c in the 
manuscript, which are then further processed and tweaked in Adobe Illustrator.
"""

import os
import numpy as np
import pandas as pd
import argparse
import json

import matplotlib as mpl
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


DEBUG = False

def _get_vertical_patch_connect_point(patch):
    """
    Extract the point on a vertical patch where a XL will connect to

    Args: patch (matplotlib.patches.Rectangle): rectangle patch object for a
    residue

    Returns: (tuple): (x, y) coordinates of the connection point
    """
    
    w = patch.get_width()
    h = patch.get_height()
    return (patch.get_x() + w/2, patch.get_y() + h/2)


def _get_rotated_patch_connect_point(patch, theta):
    """
    Extract the XL connection point on a patch that has been rotated. 

    Args: patch (matplotlib.patches.Rectangle): rectangle patch object for a
        residue

    theta (float): rotation angle in radians

    Returns: (tuple): (x, y) coordinates of the connection point
    """
    
    x0 = patch.get_x()
    y0 = patch.get_y()
    w = patch.get_width()
    h = patch.get_height()
    x = x0 + w - w*np.cos(theta) + 0.5*h*np.sin(theta)
    y = y0 + 0.5*h*np.cos(theta)
    #x = x0 + w y = y0 + h/2
    return (x, y)

    
def make_vertical_section(ax, protein, domain, struct_domains, scores, 
                          dimensions, score_settings, colormap,
                          show_unstructured=True):
    """
    Make a panel showing vertical sections. These could be the head or the arm
    (coiled-coil) regions

    Args: ax: matplotlib axes object.

    protein (str): name of the protein.

    domain (list of list): definition of the entire vertical domain. Not all of this will be structured.

    struct_domains (list of lists of lists): definitions of structured domains. These should be sorted from bottom to top i.e., in each list, the first list should be from N-terminal to C-terminal, and the second list should be in opposite order.

    scores: (list of tuples) list of (residue, score) tuples for each residue.

    score_settings: (dict) dimensions of different patches / artists etc., used for each residue.

    design: (dict) parameters for different design aspects (alpha, etc).

    colormap: pyplot colormap object ; this is a function handle that produces a color proportional value colormap(x) for a float argument x.

    show_unstructured (bool, optional): If true, this will represent the unstructured areas as thin dark lines. Defaults to True.

    Returns: dict: dict of patches (i.e. rectangles) for each residue containing
    (axes object, connection point coordinates on the patch, left or right
    arm or in the middle, structured residue or not ).
    """
    
    # dict to store info on individual residue patches
    res_patches = {}
    
    # extract the complete domain formed by the union of the different domains
    # provided
    seg_n = list(range(domain[0][0], domain[0][1]+1))
    seg_c = list(range(domain[1][0], domain[1][1]+1))
    
    # reverse the c-terminal segment
    seg_c.reverse()
    
    # get structured residues
    struct_res = []
    for d in struct_domains:
        for seg in d:
            struct_res.extend([i for i in range(seg[0], seg[1]+1)])
    
    # get dimensions
    W = dimensions["W"]
    H = dimensions["H"]
    W_UNSTRUCTURED = dimensions["FACTOR_W_UNSTRUCTURED"] * W
    
    PAD_W = dimensions["PAD_W"]
    PAD_H = dimensions["PAD_H"]
    PAD_W_UNSTRUCTURED = (W - W_UNSTRUCTURED) * dimensions["FACTOR_PAD_W_UNSTRUCTURED"]
    
    DX_INTRA = dimensions["DX_INTRA"]
     
    # set axes limits
    maxlen = max(len(seg_n), len(seg_c))
    ax.set_xlim([-PAD_W, 2*W + DX_INTRA + PAD_W])
    ax.set_ylim([-PAD_H, maxlen*H + PAD_H])
    
    # get avg. heights of domains in each arm and adjust so as to remove white
    # space
    H_SEG_N = maxlen*H / len(seg_n) 
    H_SEG_C = maxlen*H / len(seg_c)
    
    
    # -------
    # SEG-N
    # -------
    x0 = 0
    y0 = 0
    for i, r in enumerate(seg_n):
        is_struct = None
        
        if r in struct_res:
            x = x0
            y = y0 + i*H_SEG_N
            w = W
            cp_x = x + w/2
            cp_y = y + H_SEG_N/2
            score = scores[r-1][1]
            color = colormap(score)
            alpha = score_settings["FADE_ALPHA"] 
            is_struct = True
        else:
            x = x0 + PAD_W_UNSTRUCTURED
            y = y0 + i*H_SEG_N
            w = W_UNSTRUCTURED
            cp_x = x + w/2
            cp_y = y + H_SEG_N/2
            color = score_settings["UNSTRUCTURED_DOMAIN_COLOR"]
            alpha = 1.0
            is_struct = False
        
        patch = mpatches.Rectangle((x, y), w, H_SEG_N, facecolor=color, alpha=alpha)
        res_patches[(protein, r)] = (ax, (cp_x, cp_y), "left", is_struct)
        ax.add_patch(patch)
    
    # ------
    # SEG-C
    # ------   
    x0 += W + DX_INTRA
    y0 = 0
    for i, r in enumerate(seg_c):
        is_struct = None
        
        if r in struct_res:
            x = x0
            y = y0 + i*H_SEG_C
            w = W
            cp_x = x + w/2
            cp_y = y + H_SEG_C/2
            score = scores[r-1][1]
            color = colormap(score)
            alpha = score_settings["FADE_ALPHA"]
            is_struct = True
        else:
            x = x0 + PAD_W_UNSTRUCTURED
            y = y0 + i*H_SEG_C
            w = W_UNSTRUCTURED
            cp_x = x + w/2
            cp_y = y + H_SEG_C/2
            color = score_settings["UNSTRUCTURED_DOMAIN_COLOR"]
            alpha = 1.0
            is_struct = False
            
        patch = mpatches.Rectangle((x, y), w, H_SEG_C, facecolor=color, alpha=alpha)
        res_patches[(protein, r)] = (ax, (cp_x, cp_y), "right", is_struct)
        ax.add_patch(patch)
    
    # clean the axes ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    return res_patches


def make_curved_section(ax, protein, domain, scores, 
                        dimensions, score_settings, colormap):
    """
    Make a panel showing a curved section. These is the hinge region.

    Args: ax: matplotlib axes object.

    protein (str): name of the protein.

    domains (list): contains start and stop residues for the hinge region
    of either smc5 of smc6.

    scores: (list of tuples) list of (residue, score) tuples for each residue.

    dimensions: (dict) dimensions of different patches / artists etc., used in the panel.

    score_settings: (dict) dimensions of different patches / artists etc., used for each residue.

    colormap: pyplot colormap object ; this is a function handle that produces a color proportional value colormap(x) for a float argument x.

    Returns: dict: dict of patches (i.e. rectangles) for each residue containing
    (axes object, connection point coordinates on the patch, left or right
    arm or in the middle, structured residue or not ).
    """
    
    # dict to store info on individual residue patches
    res_patches = {}
    
    # extract the domain
    seg_hinge = list(range(domain[0], domain[1]))
    n_hinge = len(seg_hinge)
    
    # get dimensions
    W = dimensions["W"]
    H = dimensions["H"]
    DX_INTRA = dimensions["DX_INTRA"]
    PAD_W = dimensions["PAD_W"]
    PAD_H = dimensions["PAD_H"]
    
    # set axes limits
    ax.set_xlim([-PAD_W, 2*W + DX_INTRA + PAD_W])
    ax.set_ylim([-PAD_H, 0.5*DX_INTRA + W + PAD_H])
    
    x0 = 0
    y0 = 0
    radius = 0.5 * DX_INTRA

    is_struct = True
    
    for i, r in enumerate(seg_hinge):
        theta_deg = i*180.0 / (n_hinge-1)
        theta_rad = np.pi * theta_deg / 180.0
        
        # get rectangle edge coordinates
        x = x0 + radius*(1-np.cos(theta_rad))
        y = y0 + radius*np.sin(theta_rad)
        
        # create the patch
        score = scores[r-1][1]
        color = colormap(score)
        alpha = score_settings["FADE_ALPHA"]
        patch = mpatches.Rectangle((x,y), W, H, facecolor=color, alpha=alpha)
        
        # rotate it around the other edge
        pivot_x = patch.get_x() + patch.get_width()
        pivot_y = patch.get_y()
        rotate = Affine2D().rotate_deg_around(pivot_x, pivot_y, -theta_deg)
        patch.set_transform(rotate + ax.transData)
        
        cp_x = x + W + 0.5* H*np.sin(theta_rad)
        cp_y = y + 0.5* H*np.cos(theta_rad)
        
        res_patches[(protein, r)] = (ax, (cp_x, cp_y), "mid", is_struct)
        ax.add_patch(patch)
        
    # clean the axes ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    return res_patches


def make_intra_xl(res_patches, dimensions, xl_settings, conn_type=None):
    """
    Render intra-molecular (intra-smc5 or intra-smc6) crosslinks.

    Args: res_patches (list): list containing matplotlib.patches.Rectangle
        objects for each residue.

    dimensions: (dict) dimensions of different patches / artists etc., used in the panel.

    xl_settings (dict): contains design parameters for drawing XL connection lines, filenames for XL datasets, etc.

    conn_type (int, optional): 1-> draw only XLs between N- and C-terminal parts of a vertical section and intra-hinge XLs.
    
    2-> draw only intra-N-terminal or intra-C-terminal XLs for vertical sections, and inter-arm-hinge XLs. Defaults to None.

    Returns: (list): list of matplotlib.patches.ConnectionPatch objects
    representing the XLs.
    """
    
    conns = []
    
    # get relevant dimensions
    W = dimensions["W"]
    W_UNSTRUCTURED = dimensions["FACTOR_W_UNSTRUCTURED"] * W
    
    # get xl specific settings
    XL_FN = xl_settings["FN"]
    XL_TURN = xl_settings["TURN"]
    XL_COLOR = xl_settings["COLOR"]
    XL_LS = xl_settings["LS"]
    XL_LW = xl_settings["LW"]
    XL_ALPHA = xl_settings["ALPHA"]

    # extract the xls from the file into a tuple
    df_xl = pd.read_csv(XL_FN)
    xls = [tuple(df_xl.iloc[i]) for i in range(len(df_xl))]
    
    for xl in xls:
        # extract the XL
        p1, r1, p2, r2, linker = xl
        if not (p1 == p2):
            continue
        
        if not ((p1, r1) in res_patches and (p2, r2) in res_patches):
            continue
        
        # extract the patch details
        ax1, coord1, side1, is_struct1 = res_patches[(p1, r1)]
        ax2, coord2, side2, is_struct2 = res_patches[(p2, r2)]

        # figure out which XLs to show if a conn_type has been specified
        if conn_type == 1:
            case1 = side1 == "left" and side2 == "right"
            case2 = side1 == "right" and side2 == "left"
            case3 = side1 == "mid" and side2 == "mid"
            if not any([case1, case2, case3]):
                continue
        
        elif conn_type == 2:
            case1 = side1 == side2 == "left"
            case2 = side1 == side2 == "right"
            case3 = side1 == "left" and side2 == "mid"
            case4 = side1 == "mid" and side2 == "left"
            case5 = side1 == "right" and side2 == "mid"
            case6 = side1 == "mid" and side2 == "right"
            if not any([case1, case2, case3, case4, case5, case6]):
                continue
       
        # get width of the residues connected by this XL
        w1 = W
        w2 = W
        if not is_struct1:
            w1 = W_UNSTRUCTURED
        if not is_struct2:
            w2 = W_UNSTRUCTURED
        
        # get the turn angle of curved XL connection lines
        if abs(r2-r1) <= 4 or (side1 == "mid" or side2 == "mid"):
            TURN_ANGLE = XL_TURN * 2
        else:
            TURN_ANGLE = XL_TURN
            
        if side1 == side2 == "left":
            xyA = (coord1[0] + 0.5*w1, coord1[1])
            xyB = (coord2[0] + 0.5*w2, coord2[1])
            cstyle = mpatches.ConnectionStyle.Arc3(rad=TURN_ANGLE)
            
        elif side1 == side2 == "right":
            xyA = (coord1[0] - 0.5*w1, coord1[1])
            xyB = (coord2[0] - 0.5*w2, coord2[1])
            cstyle = mpatches.ConnectionStyle.Arc3(rad=TURN_ANGLE)
        
        elif (side1 == "mid" and side2 == "mid"):
            xyA = coord1
            xyB = coord2
            cstyle = mpatches.ConnectionStyle.Arc3(rad=TURN_ANGLE)
                    
        elif (side1 == "left" and side2 == "right") or (side1 == "right" and side2 == "left"):
            xyA = (coord1[0] + 0.5*w1, coord1[1])
            xyB = (coord2[0] - 0.5*w2, coord2[1])
            cstyle = mpatches.ConnectionStyle.Arc3(rad=0)
        
        elif (side1 == "left" and side2 == "mid"):
            xyA = (coord1[0] + 0.5*w1, coord1[1])
            xyB = coord2
            cstyle = mpatches.ConnectionStyle.Arc3(rad=TURN_ANGLE)
        
        elif (side1 == "mid" and side2 == "left"):
            xyA = coord1
            xyB = (coord2[0] + 0.5*w2, coord2[1])
            cstyle = mpatches.ConnectionStyle.Arc3(rad=TURN_ANGLE)
        
        elif (side1 == "right" and side2 == "mid"):
            xyA = (coord1[0] - 0.5*w1, coord1[1])
            xyB = coord2
            cstyle = mpatches.ConnectionStyle.Arc3(rad=TURN_ANGLE)
        
        elif (side1 == "mid" and side2 == "right"):
            xyA = coord1
            xyB = (coord2[0] - 0.5*w2, coord2[1])
            cstyle = mpatches.ConnectionStyle.Arc3(rad=TURN_ANGLE)
                
        conn = mpatches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, connectionstyle=cstyle, linestyle=XL_LS, linewidth=XL_LW, color=XL_COLOR, alpha=XL_ALPHA,zorder=5)
        
        conns.append((p1, r1, p2, r2, conn))
            
    return conns


def make_inter_xl(res_patches, dimensions, xl_settings):
    """
    Render inter-molecular (inter-smc5-smc6) crosslinks.

    Args: res_patches (list): list containing matplotlib.patches.Rectangle
        objects for each residue.

    dimensions: (dict) dimensions of different patches / artists etc., used in the panel.

    xl_settings (dict): contains design parameters for drawing XL connection lines, filenames for XL datasets, etc.

    Returns: (list): list of matplotlib.patches.ConnectionPatch objects
    representing the XLs.
    """
    
    conns = []
    
    # get relevant dimensions
    W = dimensions["W"]
    W_UNSTRUCTURED = dimensions["FACTOR_W_UNSTRUCTURED"] * W
    
    # get xl specific settings
    XL_FN = xl_settings["FN"]
    XL_TURN = xl_settings["TURN"]
    XL_COLOR = xl_settings["COLOR"]
    XL_LS = xl_settings["LS"]
    XL_LW = xl_settings["LW"]
    XL_ALPHA = xl_settings["ALPHA"]

    # extract the xls from the file into a tuple
    df_xl = pd.read_csv(XL_FN)
    xls = [tuple(df_xl.iloc[i]) for i in range(len(df_xl))]
    
    for xl in xls:
        p1, r1, p2, r2, linker = xl
        if p1 == p2:
            continue
        
        if not ((p1, r1) in res_patches and (p2, r2) in res_patches):
            continue
        
        ax1, coord1, side1, is_struct1 = res_patches[(p1, r1)]
        ax2, coord2, side2, is_struct2 = res_patches[(p2, r2)]
        
        w1 = W
        w2 = W
        if not is_struct1:
            w1 = W_UNSTRUCTURED
        if not is_struct2:
            w2 = W_UNSTRUCTURED  
        
        xyA = coord1
        xyB = coord2
        
        if side1 != "mid":
            xyA = (coord1[0] + 0.5*w1, coord1[1])
        if side2 != "mid":
            xyB = (coord2[0] - 0.5*w2, coord2[1])      
        
        cstyle = mpatches.ConnectionStyle.Arc3(rad=0)
        
        conn = mpatches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, connectionstyle=cstyle, linestyle=XL_LS, linewidth=XL_LW, color=XL_COLOR, alpha=XL_ALPHA)
        
        conns.append((p1, r1, p2, r2, conn))
    
    return conns


## MAIN ##
parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("-s", "--settings_fn", default="settings.json", help="Json file containing rendering settings and filenames, etc.")

parser.add_argument("-intra", "--intra", action="store_true", help="True to plot intra crosslinks.")

parser.add_argument("-inter", "--inter", action="store_true", help="True to plot inter crosslinks.")

parser.add_argument("-t", "--intra_conn_type", type=int, help="1 for plotting only inter-arm and intra-hinge crosslinks (all of which are intra-molecular), 2 for plotting only intra-arm and inter-arm-hinge links.")

parser.add_argument("-o", "--outprefix", default="smc56_xl", help="prefix for output files.")

parser.add_argument("-c", "--colorbar", action="store_true", help="True to render a colorbar as a separate figure")

args = parser.parse_args()
settings_fn = os.path.abspath(args.settings_fn)
show_intra = args.intra
show_inter = args.inter
intra_conn_type = args.intra_conn_type
outprefix = args.outprefix
plot_cbar = args.colorbar

# read settings from json file
with open(settings_fn, "r") as of:
    settings = json.load(of)

# parse settings
PLOTFMT = settings["PLOTFMT"]
DPI = settings["DPI"]
XL_SETTINGS = settings["XL"]
SCORE_SETTINGS = settings["SCORE"]
DOMAINS = settings["DOMAINS"] 
DIMENSIONS = settings["DIMENSIONS"]

# get smc5 scores
df_smc5 = pd.read_csv(SCORE_SETTINGS["SMC5_FN"])
smc5_scores = []
for i in range(len(df_smc5)):
    resid = df_smc5.iloc[i]["resid"]
    score = df_smc5.iloc[i][SCORE_SETTINGS["METHOD"]]
    smc5_scores.append((resid, score))
nres_smc5 = len(smc5_scores)

# get smc6 scores
df_smc6 = pd.read_csv(SCORE_SETTINGS["SMC6_FN"])
smc6_scores = []
for i in range(len(df_smc6)):
    resid = df_smc6.iloc[i]["resid"]
    score = df_smc6.iloc[i][SCORE_SETTINGS["METHOD"]]
    smc6_scores.append((resid, score))
nres_smc6 = len(smc6_scores)

# clip and normalize scores
minscore = SCORE_SETTINGS["MINSCORE"]
maxscore = SCORE_SETTINGS["MAXSCORE"]
for i, x in enumerate(smc5_scores):
    r, s = x
    if s > maxscore:
        s = maxscore
    if s < minscore:
        s = minscore
    s = (s-minscore) / (maxscore - minscore)
    smc5_scores[i] = (r, s)

for i, x in enumerate(smc6_scores):
    r, s = x
    if s > maxscore:
        s = maxscore
    if s < minscore:
        s = minscore
    s = (s-minscore) / (maxscore-minscore)
    smc6_scores[i] = (r, s)

# initialize colormap
colormap = plt.cm.get_cmap(SCORE_SETTINGS["COLORMAP"]).reversed()

# create figure and axes limits
fig = plt.figure(figsize=(DIMENSIONS["FIG_W"], DIMENSIONS["FIG_H"]))
gs = gridspec.GridSpec(DIMENSIONS["SUBPLOTS_NROWS"], 
                       DIMENSIONS["SUBPLOTS_NCOLS"], 
                       hspace=DIMENSIONS["SUBPLOTS_HSPACE"])
SMC5_COL = DIMENSIONS["SMC5_COL"]
SMC6_COL = DIMENSIONS["SMC6_COL"]

res_patches = {}
axs = []

# --------------
# SMC5
# --------------
# head
ax_smc5_head = fig.add_subplot(gs[-1, SMC5_COL])
domains = [DOMAINS["SMC5_HEAD"]]
domain = domains[0]
res_patches_ = make_vertical_section(ax=ax_smc5_head, protein="smc5",        
                                     domain=domain,
                                     struct_domains=domains, scores=smc5_scores, score_settings=SCORE_SETTINGS,dimensions=DIMENSIONS, colormap=colormap, show_unstructured=False)
res_patches.update(res_patches_)
axs.append(ax_smc5_head)

# arm
ax_smc5_arm = fig.add_subplot(gs[1:-1, SMC5_COL])
domain = DOMAINS["SMC5_ARM"]
domains = [DOMAINS["SMC5_CC1"], DOMAINS["SMC5_CC2"], DOMAINS["SMC5_CC3"]]
res_patches_ = make_vertical_section(ax=ax_smc5_arm, protein="smc5",        
                                     domain=domain,
                                     struct_domains=domains, scores=smc5_scores, score_settings=SCORE_SETTINGS, dimensions=DIMENSIONS, colormap=colormap, show_unstructured=True)
res_patches.update(res_patches_)
axs.append(ax_smc5_arm)

# hinge
ax_smc5_hinge = fig.add_subplot(gs[0, SMC5_COL])
domain = DOMAINS["SMC5_HINGE"]
res_patches_ = make_curved_section(ax=ax_smc5_hinge, protein="smc5",        
                                   domain=domain, scores=smc5_scores, dimensions=DIMENSIONS,score_settings=SCORE_SETTINGS, colormap=colormap)
res_patches.update(res_patches_)
axs.append(ax_smc5_hinge)
  
# --------------
# SMC6
# --------------
# head
ax_smc6_head = fig.add_subplot(gs[-1, SMC6_COL])
domains = [DOMAINS["SMC6_HEAD"]]
domain = domains[0]
res_patches_ = make_vertical_section(ax=ax_smc6_head, protein="smc6",        
                                     domain=domain,
                                     struct_domains=domains, scores=smc6_scores, score_settings=SCORE_SETTINGS,dimensions=DIMENSIONS, colormap=colormap, show_unstructured=False)
res_patches.update(res_patches_)
axs.append(ax_smc6_head)

# arm
ax_smc6_arm = fig.add_subplot(gs[1:-1, SMC6_COL])
domain = DOMAINS["SMC6_ARM"]
domains = [DOMAINS["SMC6_CC1"], DOMAINS["SMC6_CC2"], 
           DOMAINS["SMC6_CC3"], DOMAINS["SMC6_CC4"]]
res_patches_ = make_vertical_section(ax=ax_smc6_arm, protein="smc6",        
                                     domain=domain,
                                     struct_domains=domains, scores=smc6_scores, score_settings=SCORE_SETTINGS, dimensions=DIMENSIONS, colormap=colormap, show_unstructured=True)
res_patches.update(res_patches_)
axs.append(ax_smc6_arm)

# hinge
ax_smc6_hinge = fig.add_subplot(gs[0, SMC6_COL])
domain = DOMAINS["SMC6_HINGE"]
res_patches_ = make_curved_section(ax=ax_smc6_hinge, protein="smc6",        
                                   domain=domain, scores=smc5_scores, score_settings=SCORE_SETTINGS, dimensions=DIMENSIONS, colormap=colormap)
res_patches.update(res_patches_)
axs.append(ax_smc6_hinge)

# --------------
# CROSSLINKS
# --------------
# intra
if show_intra:
    xl_intra = make_intra_xl(res_patches, dimensions=DIMENSIONS,                
                             xl_settings=XL_SETTINGS, 
                             conn_type=intra_conn_type)
    for xl in xl_intra: 
        p1, r1, p2, r2, conn = xl
        fig.add_artist(conn)

# inter
if show_inter:
    xl_inter = make_inter_xl(res_patches, dimensions=DIMENSIONS,
                             xl_settings=XL_SETTINGS)
    for xl in xl_inter:
        p1, r1, p2, r2, conn = xl
        fig.add_artist(conn)
    
# -------
# DESIGN
# -------
# turn off all axis
for ax in axs:
    ax.axis("off")

# save
fig.subplots_adjust(hspace=DIMENSIONS["SUBPLOTS_HSPACE"],
                    wspace=DIMENSIONS["SUBPLOTS_WSPACE"])
figname = outprefix + "." + PLOTFMT
fig.savefig(figname, dpi=DPI, bbox_inches="tight")

# --------------------------
# COLORBAR (SEPARATE FIGURE)
# --------------------------
if plot_cbar:
    fig_cbar, ax_cbar = plt.subplots(1,1, figsize=(DIMENSIONS["FIG_CBAR_W"],
                                                   DIMENSIONS["FIG_CBAR_H"]))
    fig.subplots_adjust(bottom=DIMENSIONS["FIG_CBAR_BOTTOM"])
    norm = mpl.colors.Normalize(vmin=minscore, vmax=maxscore)
    cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=colormap, norm=norm,
                                   orientation="horizontal", alpha=SCORE_SETTINGS["FADE_ALPHA"])
    
    fig_cbar.tight_layout()
    #cb.set_label("conservation score")
    figname = "colorbar" + "." + PLOTFMT
    fig_cbar.savefig(figname, dpi=DPI, bbox_inches="tight")

if DEBUG:
    plt.show()