import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='ispd2005')
parser.add_argument('--output_dir', type=str, default='ispd2005_macro')
parser.add_argument('--benchmark', type=str, default='adaptec1')
args = parser.parse_args()

class PlaceDB():
    def __init__(self, benchmark, base_dir, macro_list_dir):
        self.benchmark = benchmark
        self.base_dir = os.path.join(base_dir, benchmark)
        assert os.path.exists(base_dir)
        self.node_info = self.read_node_file(macro_list_dir)
        self.net_info = self.read_net_file()
        self.read_expert_pos()

    def read_node_file(self, macro_list_dir):
        fopen = open(os.path.join(self.base_dir, self.benchmark+".nodes"), "r")
        
        node_range_list = {}
        macro_list_dir = os.path.join(macro_list_dir, self.benchmark+".pkl")
        if self.benchmark == "bigblue2" or self.benchmark == "bigblue4" or \
            self.benchmark.startswith('superblue'):
            node_range_list = pickle.load(open(macro_list_dir,'rb'))

        node_info = {}
        for line in fopen.readlines():
            if not line.startswith("\t") and not line.startswith(" "):
                continue
            line = line.strip().split()
            node_name = line[0]
            if node_range_list and node_name not in node_range_list:
                continue
            if not node_range_list and line[-1] != 'terminal':
                continue
            x, y = int(line[1]), int(line[2])
            node_info[node_name] = {"x": x , "y": y}
            
        fopen.close()
        return node_info

    def read_net_file(self):
        fopen = open(os.path.join(self.base_dir, self.benchmark+".nets"), "r")
        
        net_info = {}
        net_name = None
        for line in fopen.readlines():
            if not line.startswith("\t") and not line.startswith("  ") and \
                not line.startswith("NetDegree"):
                continue
            line = line.strip().split()
            if line[0] == "NetDegree":
                net_name = line[-1]
            else:
                node_name = line[0]
                if node_name in self.node_info:
                    if not net_name in net_info:
                        net_info[net_name] = {}
                        net_info[net_name]["nodes"] = {}
                    x_offset, y_offset = float(line[-2]), float(line[-1])
                    type = line[1]
                    net_info[net_name]["nodes"][node_name] = {"x_offset": x_offset, "y_offset": y_offset, "type": type}

        for net_name in list(net_info.keys()):
            if len(net_info[net_name]["nodes"]) <= 1:
                net_info.pop(net_name)
        
        fopen.close()
        return net_info

    def read_expert_pos(self):
        fopen = open(os.path.join(self.base_dir, self.benchmark+".pl"), "r")
        
        for line in fopen.readlines():
            line = line.strip().split()
            if len(line) < 5 or line[0].startswith('#'):
                continue
            node_name = line[0]
            if not node_name in self.node_info:
                continue
            place_x, place_y = int(line[1]), int(line[2])
            self.node_info[node_name]["raw_x"] = place_x
            self.node_info[node_name]["raw_y"] = place_y
            
        fopen.close()

def write_aux(benchmark, input_dir, output_dir):
    filer = os.path.join(input_dir, benchmark+'.aux')
    filew = os.path.join(output_dir, benchmark+'.aux')
    with open(filer, 'r') as f:
        line = f.readline()
    with open(filew, 'w') as f:
        f.write(line)

def write_wts(benchmark, input_dir, output_dir):
    filer = os.path.join(input_dir, benchmark+'.wts')
    filew = os.path.join(output_dir, benchmark+'.wts')
    if not os.path.exists(filer):
        return
    with open(filer, 'r') as fr:
        with open(filew, 'w') as fw:
            line = fr.readline()
            while line:
                fw.write(line)
                line = fr.readline()

def write_scl(benchmark, input_dir, output_dir):
    filer = os.path.join(input_dir, benchmark+'.scl')
    filew = os.path.join(output_dir, benchmark+'.scl')
    with open(filer, 'r') as fr:
        with open(filew, 'w') as fw:
            line = fr.readline()
            while line:
                if benchmark.startswith('superblue'):
                    line_tmp = line.strip()
                    if line_tmp.startswith('Height'):
                        line = f'  Height       : 3420\n'
                fw.write(line)
                line = fr.readline()

def write_nodes(benchmark, placedb:PlaceDB, output_dir):
    filew = os.path.join(output_dir, benchmark+'.nodes')
    with open(filew, 'w') as f:
        f.write('UCLA nodes 1.0\n\n')
        f.write(f'NumNodes : 		{len(placedb.node_info)}\n')
        f.write(f'NumTerminals : 		{len(placedb.node_info)}\n')
        for node in placedb.node_info:
            x, y = placedb.node_info[node]['x'], placedb.node_info[node]['y']
            f.write(f'\t{node} {int(x)}\t{int(y)}\tterminal\n')

def write_nets(benchmark, placedb:PlaceDB, output_dir):
    filew = os.path.join(output_dir, benchmark+'.nets')
    with open(filew, 'w') as f:
        f.write('UCLA nets 1.0\n\n')
        f.write(f'NumNets : {len(placedb.net_info)}\n')
        num_pins = 0
        for net in placedb.net_info:
            num_pins += len(placedb.net_info[net]['nodes'])
        f.write(f'NumPins : {num_pins}\n\n')
        for net in placedb.net_info:
            degree = len(placedb.net_info[net]['nodes'])
            f.write(f'NetDegree : {degree}   {net}\n')
            for node_name in placedb.net_info[net]['nodes']:
                x, y = placedb.net_info[net]['nodes'][node_name]['x_offset'], placedb.net_info[net]['nodes'][node_name]['y_offset']
                type = placedb.net_info[net]['nodes'][node_name]['type']
                f.write(f'\t{node_name}	{type} : {x}\t{y}\n')

def write_pl(benchmark, placedb:PlaceDB, output_dir):
    filew = os.path.join(output_dir, benchmark+'.pl')
    with open(filew, 'w') as f:
        f.write('UCLA pl 1.0\n\n')
        for node in placedb.node_info:
            x, y = int(placedb.node_info[node]['raw_x']), int(placedb.node_info[node]['raw_y'])
            f.write(f'{node}\t{x}\t{y}\t: N\n')

if __name__ == '__main__':
    placedb = PlaceDB(args.benchmark, args.input_dir, 'macro_list')
    
    input_dir = os.path.join(args.input_dir, args.benchmark)
    output_dir = os.path.join(args.output_dir, args.benchmark)
    os.makedirs(output_dir, exist_ok=True)
    
    write_aux(args.benchmark, input_dir, output_dir)
    write_wts(args.benchmark, input_dir, output_dir)
    write_scl(args.benchmark, input_dir, output_dir)
    write_nodes(args.benchmark, placedb, output_dir)
    write_nets(args.benchmark, placedb, output_dir)
    write_pl(args.benchmark, placedb, output_dir)
    
    print(f'Succesfully created macro dataset {output_dir}')