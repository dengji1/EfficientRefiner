from placedb import RefineDB
import os
import numpy as np
import time
from refiner.overlap_cuda import get_overlap, get_exact_overlap
from refiner.overlap_prun_cuda import get_overlap as get_overlap_prun, get_exact_overlap as get_exact_overlap_prun, get_big_modules
from refiner.wl_cuda import get_wl, get_hpwl
from refiner.refiner_args import Args
import torch
from refiner.modify import *

class Log:
    def __init__(self, args):
        file = os.path.join(args.output_dir, f'{args.benchmark}/{args.benchmark}.csv')
        self.fopen = open(file, 'w')
        self.fopen.write('iter,wl,overlap,ref\n')
        self.fopen.flush()
        
    def write_line(self, iter, wl, overlap,reg):
        self.fopen.write(f'{iter},{wl:.2f},{overlap:.4f},{reg:.2f}\n')
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

def get_pos_ratio(pos_init, chip:RefineDB):
    pos = (pos_init) / (chip.chip_size.reshape(-1, 2) - chip.node_size)
    pos = np.minimum(pos, 1)
    return inv_sigmoid(pos)

def get_pos_bound(chip:RefineDB):
    pos_min, pos_max = np.zeros([chip.node_cnt,2]), np.ones([chip.node_cnt,2])
    pos_min += chip.node_size / chip.chip_size.reshape(-1, 2) / 2
    pos_max -= chip.node_size / chip.chip_size.reshape(-1, 2) / 2
    return pos_min, pos_max

def refiner(chip: RefineDB, args):
    ref_args = Args(args)
    
    if args.save_curve:
        curve_log = Log(args)
    
    if args.pruning:
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
    
    # hpwl = get_hpwl(chip, pos * chip.chip_size.reshape(-1, 2), ref_args)
    # print('Initial HPWL =', hpwl)
            
    print('-----Refinement start-----')
    start_time = time.time()
    
    for it in range(1, args.iter+1):
        pos_ratio = pos_ratio_tensor.detach().numpy()
        pos = pos_min + sigmoid(pos_ratio) * (pos_max - pos_min)
        
        wl, d_pos_wl = get_wl(chip, pos, ref_args)
        d_ratio_wl = d_pos_wl * (pos_max - pos_min) * d_sigmoid(pos_ratio)
        
        if not args.pruning:
            overlap, d_pos_overlap = get_overlap(chip, pos, ref_args, chip.fix_mask)
        else:
            overlap, d_pos_overlap = get_overlap_prun(chip, pos, big_modules, bin_num, ref_args, chip.fix_mask)
        d_ratio_overlap = d_pos_overlap * (pos_max - pos_min) * d_sigmoid(pos_ratio)

        reg, d_pos_reg = get_regulate(chip, pos, ref_args)
        d_ratio_reg = d_pos_reg * (pos_max - pos_min) * d_sigmoid(pos_ratio)
        
        loss = wl + args.wo * overlap + args.wr * reg
        tot_grad = torch.tensor(d_ratio_wl + args.wo * d_ratio_overlap + args.wr * d_ratio_reg)

        fix_mask = np.zeros((chip.node_cnt, 2), dtype=np.bool_)
        fix_mask[:, 0] = fix_mask[:, 1] = chip.fix_mask
        fix_mask = torch.from_numpy(fix_mask)
        tot_grad.masked_fill_(fix_mask, 0)
        
        pos_ratio_tensor.grad = tot_grad
        optimizer.step()
        
        if it % args.print_interval == 0:
            print(f'iter={it}, loss={loss}, wl={wl}, overlap={overlap}, reg={reg}')
        
        if args.save_curve:
            curve_log.write_line(it, wl, overlap, reg)
    
    print('-----Refinement end-----')
    print(f'Refinement time: {time.time()-start_time}s')
    
    if args.save_curve:
        curve_log.close()
        
    placement = pos * chip.chip_size.reshape(-1, 2) - chip.node_size / 2
    placement[:, 0] = np.clip(placement[:, 0], 0, chip.chip_size[0])
    placement[:, 1] = np.clip(placement[:, 1], 0, chip.chip_size[1])
        
    node_pos = {}
    for id, node in enumerate(chip.id2name_node):
        node_pos[node] = {'x':placement[id, 0], 'y':placement[id, 1]}
    return node_pos
