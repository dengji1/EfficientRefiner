from re import S

from sympy import true
import legalization.dreamplace.dreamplace.PlaceDB as PlaceDB
import legalization.dreamplace.dreamplace.Params as Params
import legalization.dreamplace.dreamplace.configure as configure
import legalization.dreamplace.dreamplace.NonLinearPlace as NonLinearPlace
import numpy as np
import os
import shutil

# Convert to DreamPlace-supported format
def name_change(name:str):
    name=name.replace("_","left0right")
    name=name.replace("\[","left1right")
    name=name.replace("\]","left2right")
    name=name.replace("/","left3right")
    name=name.replace("[","left4right")
    name=name.replace("]","left5right")
    name=name.replace(".","left6right")
    name=name.replace("$","left7right")
    return name

def name_change_inv(name:str):
    name=name.replace("left7right","$")
    name=name.replace("left6right",".")
    name=name.replace("left5right","]")
    name=name.replace("left4right","[")
    name=name.replace("left3right","/")
    name=name.replace("left2right","\]")
    name=name.replace("left1right","\[")
    name=name.replace("left0right","_")
    return name

def transfer_dataset(src_dir, dst_dir, benchmark, openroad):
    os.makedirs(dst_dir, exist_ok=True)

    # scl
    src_file = os.path.join(src_dir, f'{benchmark}.scl')
    dst_file = os.path.join(dst_dir, f'{benchmark}.scl')
    shutil.copy2(src_file, dst_file)

    # nodes
    src_file = os.path.join(src_dir, f'{benchmark}.nodes')
    dst_file = os.path.join(dst_dir, f'{benchmark}.nodes')
    with open(src_file, 'r') as rf:
        with open(dst_file, 'w') as wf:
            for line in rf:
                node_line = line.split()
                if not node_line:
                    wf.write(line)
                    continue
                if line.startswith('\t') or line.startswith(' '):
                    if openroad:
                        line = f'\t{name_change(node_line[0])}'
                    else:
                        line = f'\t{node_line[0]}'
                    for t in node_line[1:]:
                        line += f'\t{t}'
                    line += '\n'
                wf.write(line)

    # nets
    src_file = os.path.join(src_dir, f'{benchmark}.nets')
    dst_file = os.path.join(dst_dir, f'{benchmark}.nets')
    with open(src_file, 'r') as rf:
        with open(dst_file, 'w') as wf:
            for line in rf:
                node_line = line.split()
                if not node_line:
                    wf.write(line)
                    continue
                if line.startswith('\t') or line.startswith(' '):
                    line = f'\t{name_change(node_line[0])}' if openroad else f'\t{node_line[0]}'
                    for t in node_line[1:]:
                        line += f'\t{t}'
                    line += '\n'
                if line.startswith('NetDegree'):
                    line = 'NetDegree'
                    for t in node_line[1:-1]:
                        line += f'\t{t}'
                    line += f'\t{name_change(node_line[-1])}\n' if openroad else f'\t{node_line[-1]}\n'
                wf.write(line)

class DreamPlaceWrapper:
    def __init__(self, benchmark, base_dir, placedb, args):
        self.params = Params.Params()
        param_pth = 'legalization/params.json'
        self.params.load(param_pth)
        self.benchmark = benchmark
        os.environ["OMP_NUM_THREADS"] = "%d" % (self.params.num_threads)
        
        if base_dir.endswith(args.macro_suffix):
            self.macro_dir = base_dir
            self.mix_dir = base_dir[:-len(args.macro_suffix)]
        else:
            self.mix_dir = base_dir
            self.macro_dir = base_dir + args.macro_suffix
        
        self.placedb = placedb
        
        self.tmp_dir = 'tmpDB'
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.args = args
    
    def clear(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
            
    def legalize(self, node_pos):
        
        def write_pl(placedb, pl_dir, node_pos, ref_to_dp, openroad):
            with open(pl_dir, 'w') as f:
                f.write('UCLA pl 1.0\n\n')
                for node_name in placedb.node_info_all:
                    orin = placedb.node_info_all[node_name]['orign']
                    
                    node_name_new = name_change(node_name) if ref_to_dp and openroad else node_name
                        
                    if node_name in node_pos:
                        orin = node_pos[node_name].get('orign', orin)
                        if placedb.node_info[node_name]['fix']:
                            x = int(placedb.expert_pos[node_name]['x'] + placedb.offset_x)
                            y = int(placedb.expert_pos[node_name]['y'] + placedb.offset_y)
                            f.write('{}\t{}\t{}\t:\t{} /FIXED_NI\n'.format(node_name_new, x, y, orin))
                        else:
                            x = int(node_pos[node_name]['x'] + placedb.offset_x)
                            y = int(node_pos[node_name]['y'] + placedb.offset_y)
                            if not ref_to_dp:
                                f.write('{}\t{}\t{}\t:\t{} /FIXED\n'.format(node_name_new, x, y, orin))
                            else:
                                f.write('{}\t{}\t{}\t:\t{}\n'.format(node_name_new, x, y, orin))
                    elif not ref_to_dp:
                        f.write('{}\t{}\t{}\t:\t{}\n'.format(node_name_new, 0, 0, orin))
        
        # write dataset
        dst_dir = os.path.join(self.tmp_dir, self.benchmark)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        src_dir = os.path.join(self.macro_dir, self.benchmark)
        transfer_dataset(src_dir, dst_dir, self.benchmark, self.placedb.openroad)
        
        pl_dir = os.path.join(dst_dir, f'{self.benchmark}.refine.pl')
        write_pl(self.placedb, pl_dir, node_pos, True, self.placedb.openroad)
        
        aux_dir = os.path.join(dst_dir, f'{self.benchmark}.refine.aux')
        with open(aux_dir, 'w') as f:
            f.write(f'RowBasedPlacement : {self.benchmark}.nodes {self.benchmark}.nets {self.benchmark}.refine.pl {self.benchmark}.scl\n')
        
        # configure params
        lg_params = self.params
        lg_params.aux_input = aux_dir
        lg_params.global_place_flag = 0
        lg_params.legalize_flag = 1
        lg_params.detailed_place_flag = 0
        tmp_pl_dir = os.path.join(dst_dir, f'{self.benchmark}.lg.pl')
        
        # legalize
        self.place(lg_params, tmp_pl_dir)
        
        # write to pl
        node_pos_lg = {}
        with open(tmp_pl_dir, 'r') as f:
            for line in f:
                pos_line = line.strip().split()
                if len(pos_line) >= 4 and not line.startswith('#'):
                    node_name = name_change_inv(pos_line[0]) if self.placedb.openroad else pos_line[0]
                    node_pos_lg[node_name] = {}
                    node_pos_lg[node_name]['x'] = float(pos_line[1]) - self.placedb.offset_x
                    node_pos_lg[node_name]['y'] = float(pos_line[2]) - self.placedb.offset_y
                    node_pos_lg[node_name]['orign'] = pos_line[4]
        pl_dir = os.path.join(self.args.output_dir, 
                              self.args.benchmark, f'{self.benchmark}.lg.pl')

        write_pl(self.placedb, pl_dir, node_pos_lg, False, self.placedb.openroad)
        return node_pos_lg
    
    def place_cells(self):
        # prepare files
        dst_dir = os.path.join(self.tmp_dir, self.benchmark+'-mix')
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        src_dir = os.path.join(self.mix_dir, self.benchmark)
        transfer_dataset(src_dir, dst_dir, self.benchmark, self.placedb.openroad)
        
        src_pl = os.path.join(self.args.output_dir, 
                              self.args.benchmark, f'{self.benchmark}.lg.pl')
        dst_pl = os.path.join(dst_dir, f'{self.benchmark}-mix.lg.pl')
        with open(src_pl, 'r') as rf:
            with open(dst_pl, 'w') as wf:
                for line in rf:
                    parts = line.split()
                    if len(parts) > 4:
                        if self.placedb.openroad:
                            line = f'\t{name_change(parts[0])}'
                        else:
                            line = f'\t{parts[0]}'
                        for t in parts[1:]:
                            line += f'\t{t}'
                        line += '\n'
                    wf.write(line)
        
        aux_dir = os.path.join(dst_dir, f'{self.benchmark}.lg.aux')
        with open(aux_dir, 'w') as f:
            f.write(f'RowBasedPlacement : {self.benchmark}.nodes {self.benchmark}.nets {self.benchmark}-mix.lg.pl {self.benchmark}.scl\n')
        
        # configure params
        dp_params = self.params
        dp_params.aux_input = aux_dir
        dp_params.global_place_flag = 1
        dp_params.legalize_flag = 1
        dp_params.detailed_place_flag = 1
        if self.placedb.benchmark.startswith('superblue'):
            dp_params.detailed_place_flag = 0
        src_pl = os.path.join(dst_dir, f'{self.benchmark}.dp.pl')
        dst_pl = os.path.join(self.args.output_dir, 
                              self.args.benchmark, f'{self.benchmark}.dp.pl')
        
        # full placement
        self.place(dp_params, src_pl)
        
        # write pl
        with open(src_pl, 'r') as rf:
            with open(dst_pl, 'w') as wf:
                for line in rf:
                    if self.placedb.openroad:
                        line1 = name_change_inv(line)
                    else:
                        line1 = line
                    wf.write(line1)
    
    def place(self, params, pl_dir):
        assert (not params.gpu) or configure.compile_configurations["CUDA_FOUND"] == 'TRUE', \
            "CANNOT enable GPU without CUDA compiled"
        np.random.seed(params.random_seed)
        
        placedb = PlaceDB.PlaceDB()
        placedb(params)
        
        placer = NonLinearPlace.NonLinearPlace(params, placedb, None)
        metrics = placer(params, placedb)
        placedb.write(params, pl_dir)
