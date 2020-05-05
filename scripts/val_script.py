import sys
import os
import argparse
import io
import pandas as pd
sys.path.append('/home/rishi/py_scripts')
sys.path.append('/home/rgur/py_scripts')
import rishi_utils as ru

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help='directory containing VJTNN(s) of interest')
parser.add_argument('--num', type=int, help='number of models to test')
parser.add_argument("--n_decode", type=int, help="number of translations per source molecule")
parser.add_argument('--data_dir', type=str, help='path to data directory', required=True)
parser.add_argument('--iclr_dir', type=str, help='path to parent directory of ICLR')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch to start decoding from')
parser.add_argument('--sim_delta', type=float, default=0.2, help='similarity threshold')
parser.add_argument("--predictors", type=str, help="list of python files to run for property evaluation", nargs='+') 
parser.add_argument("--prop_targets", type=str, help="list of property targets", nargs='+', required=True)
parser.add_argument('--decode', type=str, help='should polymers be decoded?')
#parser.add_argument("--prop_targets", type=list, help="list of property targets")


args = parser.parse_args()
DIR = args.dir
NUM = args.num
N_DECODE = args.n_decode
DATA_DIR = args.data_dir
ICLR_DIR = args.iclr_dir
start_epoch = args.start_epoch
SIM_DELTA = args.sim_delta
args.prop_targets = ru.pass_argparse_list(args.prop_targets)
args.decode = ru.str2bool(args.decode)
# args = sys.argv

# DIR=args[1] #model directory
# NUM=int(args[2]) #number of models to test
# N_DECODE=args[3] #number of monomers per test monomers #number of test monomers
# BG_PATH=args[4] #path to bandgap predictor
# DATA_DIR=args[5] #path to data
# ICLR_DIR=args[6] #path to iclr directory
# try:
#     start_epoch=int(args[7])
# except:
#     start_epoch = 0
# try:
#     SIM_DELTA=float(args[8])
# except:
#     SIM_DELTA =.2
    
if DIR[-1] == '/':
    DIR = DIR[:-1]

def tail( f, lines=20 ):
    total_lines_wanted = lines

    BLOCK_SIZE = 1024
    f.seek(0, 2)
    block_end_byte = f.tell()
    lines_to_go = total_lines_wanted
    block_number = -1
    blocks = [] # blocks of size BLOCK_SIZE, in reverse order starting
                # from the end of the file
    while lines_to_go > 0 and block_end_byte > 0:
        if (block_end_byte - BLOCK_SIZE > 0):
            # read the last block we haven't yet read
            f.seek(block_number*BLOCK_SIZE, 2)
            blocks.append(f.read(BLOCK_SIZE))
        else:
            # file too small, start from begining
            f.seek(0,0)
            # only read what was not read
            blocks.append(f.read(block_end_byte))
        lines_found = blocks[-1].count('\n')
        lines_to_go -= lines_found
        block_end_byte -= BLOCK_SIZE
        block_number -= 1
    all_read_text = ''.join(reversed(blocks))
    return '\n'.join(all_read_text.splitlines()[-total_lines_wanted:])

with open('%stest.txt' %args.data_dir, 'r') as f:
    lines = f.readlines()
    total_n = len([l for l in lines if l.strip(' \n') != ''])

max_acc = 0.0
best_epoch_acc = 0
max_div = 0.0
best_epoch_div = 0

for i in range(NUM):
    i += start_epoch
    f="%s/model.iter-%s" %(DIR, str(i))
    print(f)
    if os.path.isfile(f):
        #decode polymers
        if args.decode:
            os.system('python %s/iclr19-graph2graph/diff_vae/decode.py --num_decode %s --test %stest.txt --vocab %svocab.txt --model %s --use_molatt > decoded_polymers.%s' %(ICLR_DIR, N_DECODE, DATA_DIR, DATA_DIR, f, str(i)))
        #polymerize molecules
        os.system( 'python %s/iclr19-graph2graph/scripts/polymerize.py < decoded_polymers.%s > polymers.%s' %(ICLR_DIR,i,i) )
        #score decoded polymers
        os.system('rm decoded_polymers.%s' %i)
        os.system('python %s/iclr19-graph2graph/scripts/score.py --predictors %s < polymers.%s > results.%s' %(ICLR_DIR, args.predictors, str(i), str(i)))
        #analyze scored polymers       
        os.system('python %s/iclr19-graph2graph/scripts/analyze.py --num_decode %s --sim_delta %s --prop_targets %s --total_n %s --mols_path %smols.txt < results.%s > analyze.%s' %(ICLR_DIR, N_DECODE, SIM_DELTA, args.prop_targets, 
                                                                         total_n, DATA_DIR, str(i), str(i)) )  
        with open('analyze.%s' %(str(i)) ) as f:
            lines = tail(f, 2)

        split_lines = lines.split('\n')
        acc = float(split_lines[0].split()[2])
        div = float(split_lines[1].split()[2])

        if acc > max_acc:
            max_acc = acc
            best_epoch_acc = i
        if div > max_div:
            max_div = div
            best_epoch_div = i

print "Epoch with best model: ", best_epoch_acc
print "Accuracy for best model: ", max_acc

print "Epoch with most diverse model: ", best_epoch_div
print "Diversity of that model: ", max_div
