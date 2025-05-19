from placedb import PlaceDB, RefineDB
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from refiner.refiner import refiner
import argparse
import os
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=50000, help='number of refinement iterations')
parser.add_argument('--benchmark_dir', type=str, default='benchmark/ispd2005_macro')
parser.add_argument('--benchmark', type=str, default='adaptec1')
parser.add_argument('--mix', action='store_true', default=False)
parser.add_argument('--pl_dir', type=str, default='input_pl')
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('--save_pl', action='store_true', default=True)
parser.add_argument('--save_plot', action='store_true', default=False)
parser.add_argument('--save_curve', action='store_true', default=False)
parser.add_argument('--print_interval', type=int, default=500, help='interval for printing refinment information')
parser.add_argument('--save_curve_interval', type=int, default=1, help='interval for saving HPWL and overlap')
parser.add_argument('--legalization_interval', type=int, default=200, help='interval for legalization')
parser.add_argument('--alpha', type=float, default=1e5, help='hyperparameter for balancing the HPWL and Overlap objectives')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=1e-5, help='hyperparameter in the weighted average approximation formulation of HPWL')
args = parser.parse_args()

def save_plot(node_pos, placedb:PlaceDB, plot_dir, before_ref=True):
    if before_ref:
        plot_dir = os.path.join(plot_dir, 'before_refine.png')
    else:
        plot_dir = os.path.join(plot_dir, 'after_refine.png')
    
    ratio = 8 / max(placedb.width, placedb.height)
    fig_width = placedb.width * ratio
    fig_height = placedb.height * ratio
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    for node_name in node_pos:
        x, y = node_pos[node_name]['x'], node_pos[node_name]['y'] 
        size_x, size_y = placedb.node_info[node_name]['x'], placedb.node_info[node_name]['y']
        ax.add_patch(
            patches.Rectangle(
                (x*ratio, y*ratio), 
                size_x*ratio, size_y*ratio,
                linewidth=1, edgecolor='k', facecolor='b'
            )
        )
        
    ax.set_xlim([0, fig_width])
    ax.set_ylim([0, fig_height])
        
    fig.savefig(plot_dir, dpi=90, bbox_inches='tight')
    print(f'Placement plot saved to {plot_dir}')
    plt.close()
    
def save_pl(benchmark, node_pos, pl_dir):
    pl_dir = os.path.join(pl_dir, benchmark+'.pl')
    with open(pl_dir, 'w') as fwrite:
        fwrite.write('UCLA pl 1.0\n\n')
        for node_name in node_pos:
            x, y = int(node_pos[node_name]['x']), int(node_pos[node_name]['y'])
            fwrite.write('{}\t{}\t{}\t:\tN\n'.format(node_name, x, y))
    print(f'Pl file saved to {pl_dir}')

if __name__ == '__main__':
    placedb = PlaceDB(args.benchmark, args.benchmark_dir, args.pl_dir)
    refinedb = RefineDB(placedb)
    
    if args.save_plot:
        plot_dir = os.path.join(args.output_dir, args.benchmark)
        os.makedirs(plot_dir, exist_ok=True)
        save_plot(placedb.expert_pos, placedb, plot_dir, before_ref=True)
    
    ref_pos = refiner(refinedb, args)
    
    if args.save_plot:
        plot_dir = os.path.join(args.output_dir, args.benchmark)
        os.makedirs(plot_dir, exist_ok=True)
        save_plot(ref_pos, placedb, plot_dir, before_ref=False)
    
    if args.save_pl:
        pl_dir = os.path.join(args.output_dir, args.benchmark)
        os.makedirs(pl_dir, exist_ok=True)
        save_pl(args.benchmark, ref_pos, pl_dir)