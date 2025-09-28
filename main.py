from placedb import PlaceDB, RefineDB
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from refiner.refiner import refiner
import argparse
import os
import warnings
from datetime import datetime
from numba.core.errors import NumbaPerformanceWarning
from legalization.lg_dp import DreamPlaceWrapper
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, default=5000, help='number of refinement iterations')
parser.add_argument('--benchmark_dir', type=str, default='benchmark/iccad2015')
parser.add_argument('--benchmark', type=str, default='superblue3')
parser.add_argument('--pl_path', type=str, default='input_pl/superblue3.pl')
parser.add_argument('--output_dir', type=str, default='results')
parser.add_argument('--pruning', action='store_true', default=False, help='enable pruning')
parser.add_argument('--save_plot', action='store_true', default=True)
parser.add_argument('--save_curve', action='store_true', default=True)
parser.add_argument('--print_interval', type=int, default=500, help='interval for printing refinment information')
parser.add_argument('--save_curve_interval', type=int, default=1, help='interval for saving HPWL and overlap')
parser.add_argument('--wo', type=float, default=1e5, help='weight for overlap')
parser.add_argument('--wr', type=float, default=2, help='weight for regularity')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=5e-5, help='hyperparameter in the weighted average approximation formulation of HPWL')
parser.add_argument('--legalize', action='store_true', default=True, help='whether to legalize macro placement')
parser.add_argument('--place_cells', action='store_true', default=True, help='whether to place standard cells')
parser.add_argument('--macro_suffix', type=str, default='_512macro', help='suffix for macro files') #
args = parser.parse_args()

def save_plot(node_pos, placedb:PlaceDB, plot_dir, before_ref=True, lg=False):
    if before_ref:
        plot_dir = os.path.join(plot_dir, 'before_refine.png')
    elif lg:
        plot_dir = os.path.join(plot_dir, 'after_lg.png')
    else:
        plot_dir = os.path.join(plot_dir, 'before_lg.png')
    
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
    
def save_pl(benchmark, node_pos, pl_dir, placedb:PlaceDB):
    pl_dir = os.path.join(pl_dir, benchmark+'.pl')
    with open(pl_dir, 'w') as fwrite:
        fwrite.write('UCLA pl 1.0\n\n')
        for node_name in placedb.node_info_all:
            orin = placedb.placedb.node_info_all[node_name]['orign']
            if node_name in node_pos:
                if placedb.node_info[node_name]['fix']:
                    x = int(placedb.expert_pos[node_name]['x'] + placedb.offset_x)
                    y = int(placedb.expert_pos[node_name]['y'] + placedb.offset_y)
                    fwrite.write('{}\t{}\t{}\t:\t{} /FIXED_NI\n'.format(node_name, x, y, orin))
                else:
                    x = int(node_pos[node_name]['x'] + placedb.offset_x)
                    y = int(node_pos[node_name]['y'] + placedb.offset_y)
                    fwrite.write('{}\t{}\t{}\t:\t{} /FIXED\n'.format(node_name, x, y, orin))
            else:
                x, y = 0, 0
                fwrite.write('{}\t{}\t{}\t:\t{}\n'.format(node_name, x, y, orin))
    print(f'Pl file saved to {pl_dir}')

if __name__ == '__main__':
    placedb = PlaceDB(args.benchmark, args.benchmark_dir, args.pl_path, args)
    refinedb = RefineDB(placedb)
    
    if args.save_plot:
        plot_dir = os.path.join(args.output_dir, args.benchmark)
        os.makedirs(plot_dir, exist_ok=True)
        save_plot(placedb.expert_pos, placedb, plot_dir, before_ref=True)
    
    ref_pos = refiner(refinedb, args)
    
    if args.save_plot:
        plot_dir = os.path.join(args.output_dir, args.benchmark)
        os.makedirs(plot_dir, exist_ok=True)
        save_plot(ref_pos, placedb, plot_dir, before_ref=False, lg=False)
    
    if not args.legalize:
        pl_dir = os.path.join(args.output_dir, args.benchmark)
        os.makedirs(pl_dir, exist_ok=True)
        save_pl(args.benchmark, ref_pos, pl_dir, placedb)
    else:
        dp_wrapper = DreamPlaceWrapper(args.benchmark, args.benchmark_dir, placedb, args)
        pl_pos = dp_wrapper.legalize(ref_pos)
        if args.save_plot:
            save_plot(pl_pos, placedb, plot_dir, before_ref=False, lg=True)
            
        if args.place_cells:
            dp_wrapper.place_cells()
        
        dp_wrapper.clear()
