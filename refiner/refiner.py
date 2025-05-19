from placedb import RefineDB
import os
import numpy as np
import time
from refiner.overlap_cuda import get_overlap, get_exact_overlap
from refiner.overlap_mix_cuda import get_overlap as get_overlap_mix, get_exact_overlap as get_exact_overlap_mix, get_big_modules
from refiner.wl_cuda import get_wl, get_hpwl
from refiner.refiner_args import Args
import torch
from legalization.legalization import MacroLegalization

class Log:
    def __init__(self, args):
        file = os.path.join(args.output_dir, f'{args.benchmark}/{args.benchmark}.csv')
        self.fopen = open(file, 'w')
        self.fopen.write('iter,hpwl,overlap\n')
        self.fopen.flush()
        
    def write_line(self, iter, hpwl, overlap):
        self.fopen.write(f'{iter},{hpwl:.2f},{overlap:.4f}\n')
        self.fopen.flush()
        
    def close(self):
        self.fopen.close()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def inv_sigmoid(x):
    x = np.clip(x, 1e-15, 1 - 1e-15)
    return -np.log(1 / x - 1)

def init_pos_ratio(chip:RefineDB):
    pos = (chip.raw_node_pos) / (chip.chip_size.reshape(-1, 2) - chip.node_size)
    pos = np.minimum(pos, 1)
    return inv_sigmoid(pos)

def get_pos_bound(chip:RefineDB):
    pos_min, pos_max = np.zeros([chip.node_cnt,2]), np.ones([chip.node_cnt,2])
    pos_min += chip.node_size / chip.chip_size.reshape(-1, 2) / 2
    pos_max -= chip.node_size / chip.chip_size.reshape(-1, 2) / 2
    return pos_min, pos_max

def refiner(chip: RefineDB, args):
    ref_args = Args(args)
    if not args.mix:
        legalizer = MacroLegalization(chip, args)
        legal_placement = np.zeros((chip.node_cnt, 2), dtype=np.float32)
    
    if args.mix:
        bin_x = np.percentile(chip.node_size[:, 0], ref_args.percentile)
        bin_y = np.percentile(chip.node_size[:, 1], ref_args.percentile)
        bin_num = int(min(chip.chip_size[0] // bin_x, chip.chip_size[1] // bin_y))
        bin_num = min(bin_num, ref_args.min_bin_num)
        big_modules = get_big_modules(chip, bin_num, ref_args)
        print(f'Placement regoin divided into {bin_num}x{bin_num} bins')
        print(f'Number of big modules: {big_modules.sum()}, number of small modules: {big_modules.shape[0] - big_modules.sum()}')

    pos_ratio = init_pos_ratio(chip)
    pos_ratio_tensor = torch.tensor(pos_ratio, requires_grad=True)
    optimizer = torch.optim.Adam([pos_ratio_tensor], lr=args.lr)
    pos_min, pos_max = get_pos_bound(chip)
    pos_ratio = pos_ratio_tensor.detach().numpy()
    
    pos = pos_min + sigmoid(pos_ratio) * (pos_max - pos_min)
    placement = pos * chip.chip_size.reshape(-1, 2) - chip.node_size / 2
    best_placement = None
    best_hpwl = np.inf
    
    hpwl = get_hpwl(chip, pos * chip.chip_size.reshape(-1, 2), ref_args)
    if args.save_curve:
        curve_log = Log(args)
        if not args.mix:
            overlap = get_exact_overlap(chip, pos, ref_args)
        else:
            overlap = get_exact_overlap_mix(chip, pos, ref_args)
        curve_log.write_line(0, hpwl, overlap)
            
    print('-----Refinement start-----')
    print('Initial HPWL =', hpwl)
    start_time = time.time()
    
    for it in range(1, args.iter+1):
        pos_ratio = pos_ratio_tensor.detach().numpy()
        pos = pos_min + sigmoid(pos_ratio) * (pos_max - pos_min)
        
        wl, d_pos_wl = get_wl(chip, pos, ref_args)
        d_ratio_wl = d_pos_wl * (pos_max - pos_min) * d_sigmoid(pos_ratio)
        
        if not args.mix:
            overlap, d_pos_overlap = get_overlap(chip, pos, ref_args)
        else:
            overlap, d_pos_overlap = get_overlap_mix(chip, pos, big_modules, bin_num, ref_args)
        d_ratio_overlap = d_pos_overlap * (pos_max - pos_min) * d_sigmoid(pos_ratio)
        
        loss = wl + args.alpha * overlap

        pos_ratio_tensor.grad = torch.tensor(d_ratio_wl + args.alpha * d_ratio_overlap)
        optimizer.step()
        
        if it % args.print_interval == 0:
            print(f'iter={it}, loss={loss}, wl={wl}, overlap={overlap}')
        
        if args.save_curve:
            if not args.mix:
                overlap = get_exact_overlap(chip, pos, ref_args)
            else:
                overlap = get_exact_overlap_mix(chip, pos, ref_args)
            curve_log.write_line(it, hpwl, overlap)

        if not args.mix and it > args.iter * 0.9 and it % args.legalization_interval == 0:
            placement = pos * chip.chip_size.reshape(-1, 2) - chip.node_size / 2
            placement, legal = legalizer.legalize(placement)
            placement = placement.detach().cpu().numpy()
            legal_placement[:, 0], legal_placement[:, 1] = placement[:chip.node_cnt], placement[chip.node_cnt:]
            hpwl = get_hpwl(chip, legal_placement + chip.node_size / 2, ref_args)
            print(f'HPWL after legalization = {hpwl}')
            if hpwl < best_hpwl and legal:
                best_hpwl = hpwl
                best_placement = legal_placement.copy()
    
    print('-----Refinement end-----')
    print(f'Refinement time: {time.time()-start_time}s')
    if not args.mix:
        print(f'Best HPWL = {best_hpwl}')
    
    if args.save_curve:
        curve_log.close()
        
    if args.mix:
        best_placement = pos * chip.chip_size.reshape(-1, 2) - chip.node_size / 2
        
    node_pos = {}
    for id, node in enumerate(chip.id2name_node):
        node_pos[node] = {'x':best_placement[id, 0], 'y':best_placement[id, 1]}
    return node_pos