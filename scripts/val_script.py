import sys
import os

args = sys.argv

DIR=args[1] #model directory
NUM=int(args[2]) #number of models to test
N_DECODE=args[3] #number of monomers per test monomers #number of test monomers
BG_PATH=args[4] #path to bandgap predictor
DATA_DIR=args[5]

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

with open('%stest.txt' %DATA_DIR, 'r') as f:
    lines = f.readlines()
    total_n = len([l for l in lines if l.strip(' \n') != ''])

max_acc = 0.0
best_epoch_acc = 0
max_div = 0.0
best_epoch_div = 0

for i in range(NUM):
    #i += 7 #comment out later
    f="%s/model.iter-%s" %(DIR, str(i))
    print(f)
    if os.path.isfile(f):
#         os.system('python ~/iclr19-graph2graph/diff_vae/decode.py --num_decode %s --test %stest.txt --vocab %svocab.txt --model %s --use_molatt | python ~/iclr19-graph2graph/scripts/bg_score.py %s > results.%s' %(N_DECODE, DATA_DIR, DATA_DIR, f, BG_PATH, str(i)))
        os.system('python ~/iclr19-graph2graph/diff_vae/decode.py --num_decode %s --test %stest.txt --vocab %svocab.txt --model %s --use_molatt > decoded_polymers.txt' %(N_DECODE, DATA_DIR, DATA_DIR, f))
        os.system('python ~/iclr19-graph2graph/scripts/bg_score.py %s < decoded_polymers.txt > results.%s' %(BG_PATH, str(i)))

        os.system('python ~/iclr19-graph2graph/scripts/bg_analyze.py --num_decode %s --sim_delta .2 --prop_delta 6 --total_n %s --mols_path %smols.txt < results.%s > analyze.%s' %(N_DECODE, total_n, DATA_DIR, str(i), str(i)) )
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
