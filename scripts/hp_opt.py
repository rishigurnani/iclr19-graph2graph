import os
import sys
from multiprocessing import Pool
import time
import argparse
from skopt import gp_minimize
import sys
sys.path.append('/home/rishi/py_scripts')
sys.path.append('/home/rgur/py_scripts')
import rishi_utils as ru
from rishi_utils import mkdir_existOk
import random
sys.setrecursionlimit(3000)

parser = argparse.ArgumentParser(description='specify test data size or percent')
parser.add_argument("--lr_grid", type=float,
                    help="list of learning rates to try out", nargs='+', default = [.001])
parser.add_argument("--bs_grid", type=int,
                    help="list of batch sizes to try out", nargs='+', default=[32])
parser.add_argument("--dT_grid", type=int,
                    help="list of depthT values to try out", nargs='+', default=[6])

parser.add_argument("--dG_grid", type=int,
                    help="list of depthT values to try out", nargs='+', default=[8])
parser.add_argument('--train', action='store_true', help='should any models be trained?')
parser.add_argument("--hs_grid", type=int,
                    help="list of hidden layer node values to try out", nargs='+', default=[300])
parser.add_argument("--epochs", type=int,
                    help="number of epochs", default=3)
parser.add_argument("--n_decode", type=int,
                    help="number of translated polymers per source polymer", default=2)
parser.add_argument('--bayesian', action='store_true', help='should hyperparameter optimization be bayesian?')
parser.add_argument("--iclr_dir", type=str,
                    help="directory containing repository", default='/home/rgur/g2g_new')
parser.add_argument("--n_calls", type=int,
                    help="number of points in hp space to sample", default=10)
parser.add_argument("--sim_delta", type=float, help='similarity threshold', default=.2)
parser.add_argument("--predictors", type=str, help="list of relative paths to python files to run for property evaluation", nargs='+', required=True)
parser.add_argument("--prop_targets", type=str, help="list of property targets", nargs='+', required=True)
parser.add_argument('--decode', action='store_true', help='should polymers be decoded?')


args = parser.parse_args()

skip = [] #add hyperparameter combos which should be skipped

params = {'lr': args.lr_grid,
         'batch_size': args.bs_grid,
          'depthT': args.dT_grid,
          'depthG': args.dG_grid,
          'hidden_size': args.hs_grid
         }

# params = {'lr': [.0002],
#          'batch_size': [128],
#           'depthT': [6],
#           'depthG': [3]
#          }

epochs = args.epochs

n_decode = args.n_decode #default is 10

best_results = []
diversity_results = []

cwd = os.getcwd() + '/'

args.predictors = [cwd+path for path in args.predictors] #turn relative path into absolute path
args.prop_targets = ru.pass_argparse_list(args.prop_targets)

round_tup = []

start = time.time()

# space = [(200, 800), #hidden_size
#          (32, 400), #batch_size
#          (8, 100), #rand_size
#          (3, 15), #depthT
#          (3, 15), #depthG
#          (.5, .9) #anneal_rate
#          (.005, .002) #learning_rate
# ]

space = [tuple(args.hs_grid), #hidden_size
         tuple(args.bs_grid), #batch_size
         tuple(args.dT_grid), #depthT
         tuple(args.dG_grid), #depthG
         tuple(args.lr_grid) #learning_rate
]


data_dir = cwd + 'data/'
print data_dir
processed_dir = cwd + 'processed/'
test_frac = .2
train_frac = .6
SIM_DELTA = .075
#N_THREADS = 4

def train_model(hidden_size, batch_size, depthT, depthG, lr, models_dir):
    os.system('python %s/iclr19-graph2graph/diff_vae/vae_train.py --train %s --vocab %svocab.txt --save_dir %s \
                --hidden_size %s --rand_size 16 --epoch %s --anneal_rate 0.8 --lr %s --batch_size %s --depthT %s --depthG %s | \
              tee %s/LOG' %(args.iclr_dir,processed_dir, data_dir, models_dir, hidden_size, epochs, lr, batch_size, depthT, 
              depthG, models_dir) )

def validate(path, models_dir):
    hp_n_decode = 3
    hp_n_epoch = '1'
    start_epoch = 2
    os.system('python %s/iclr19-graph2graph/scripts/val_script.py %s %s %s %s %s %s %s %s > %sbest_of_round.txt' 
              %(args.iclr_dir, models_dir, hp_n_epoch, hp_n_decode, bg_path, data_dir, args.iclr_dir, start_epoch, 
                SIM_DELTA, path) )
    n_epoch = 0
    best_acc = 0.0
    div_epoch = 0
    best_div = 0
    with open('%sbest_of_round.txt' %path, 'r') as f:
        for line in f:
            l = line.strip()
            if 'Epoch with best model' in l:
                n_epoch = l.split(': ')[1]
            elif 'Accuracy for best model' in l:    
                best_acc  = l.split(': ')[1]
            elif 'Epoch with most diverse model' in l:
                div_epoch = l.split(': ')[1]
            elif 'Diversity of that model' in l:
                best_div = l.split(': ')[1]

    best_results.append((path, n_epoch, best_acc))
    diversity_results.append((path, div_epoch, best_div))
    
    srt_acc = sorted(best_results, key=lambda x: x[2], reverse=True)
    srt_div = sorted(diversity_results, key=lambda x: x[2], reverse=True)
    #print("Acc %s" %srt_acc[0][2])
    return float(srt_acc[0][2]) #top accuracy    

def objective(params):
    round_start = time.time()
    hidden_size = params[0]
    batch_size = params[1]
    depthT = params[2]
    depthG = params[3]
    lr = params[4]
    
    path = cwd + 'lr_%s_bs_%s_depthT_%s_depthG_%s_hs_%s/' %(lr, batch_size, depthT, depthG, hidden_size)
    print "Starting lr_%s_bs_%s_depthT_%s_depthG_%s_hs_%s" %(lr, batch_size, depthT, depthG, hidden_size)
    models_dir = path+'newmodels'
    results_dir = path+'results/'
    
    #make dirs if they don't exist
    if not os.path.isdir(path):
        os.makedirs(path)
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    os.chdir(results_dir)
    
    train_model(hidden_size, batch_size, depthT, depthG, lr, models_dir, results_dir)  
    top_acc = validate(path, models_dir)
    
    round_end = time.time()
    print '\n'
    print 'Finished lr_%s_bs_%s_depthT_%s_depthG_%s' %(lr, batch_size, depthT, depthG)
    print 'Time to complete round: %s' %(round_end - round_start)
    print 'Total elapsed time: %s' %(round_end - start)
    
    return -top_acc #lower is better

def make_hp_data():

    keep_test = []
    with open('%stest.txt' %data_dir, 'r') as handle:
        for line in handle:
            if random.random() <= test_frac:
                keep_test.append(line.split()[0])
    print "Testing on %s polymers" %( len(keep_test) )
    
    keep_train = []
    with open('%strain.txt' %data_dir, 'r') as handle:
        for line in handle:
            if random.random() <= train_frac:
                keep_train.append(' '.join(line.split()))
    print "Training on %s pairs" %( len(keep_train) )

    mkdir_existOk(hp_data_dir)
    ru.write_list_to_file(keep_test, '%stest.txt' %hp_data_dir)
    ru.write_list_to_file(keep_train, '%strain.txt' %hp_data_dir)
   
    nested_train = [duo.split() for duo in keep_train]
    flat_train = []
    for i in nested_train:
        flat_train.extend(i)
    flat_train = set(flat_train)
    keep_mols = flat_train.union(set(keep_test))
    ru.write_list_to_file(keep_mols, '%smols.txt' %hp_data_dir)

    os.system('python %siclr19-graph2graph/scripts/preprocess.py --train %strain.txt --mols_per_pkl 400 \
    --ncpu 8' %(args.iclr_dir, hp_data_dir) )
    hp_processed_dir = '%sprocessed/' %hp_data_dir
    mkdir_existOk(hp_processed_dir)
    os.system('mv tensor* %s' %hp_processed_dir)
    os.system('python %siclr19-graph2graph/fast_jtnn/mol_tree.py < %smols.txt > %svocab.txt' %
              (args.iclr_dir, hp_data_dir, hp_data_dir) )

if args.bayesian:
    print "Bayesian not yet supported"
#     hp_data_dir = data_dir[:-1] + '_hp' + '/'
#     make_hp_data()
#     data_dir = hp_data_dir
#     processed_dir = '%sprocessed/' %hp_data_dir 
#     r = gp_minimize(objective, space, n_calls=args.n_calls)
#     print "Optimized HPs are %s" %r.x

else:
    if args.train:
        for lr in params['lr']:
            for batch_size in params['batch_size']:
                for depthT in params['depthT']:
                    for depthG in params['depthG']:
                        for hidden_size in params['hidden_size']:
                            round_name='lr_%s_bs_%s_depthT_%s_depthG_%s_hs_%s' %(lr, batch_size, depthT, depthG, hidden_size)
                            path = cwd + round_name + '/'
                            results_dir = path+'results/'
                            if (lr, batch_size, depthT, depthG, hidden_size) not in skip:
                                round_start = time.time()
                                print '\n'
                                print 'Starting %s' %round_name
                                print'\n'
                                models_dir = path+'newmodels'

                                if not os.path.isdir(path):
                                    os.makedirs(path)
                                if not os.path.isdir(models_dir):
                                    os.makedirs(models_dir)
                                if not os.path.isdir(results_dir):
                                    os.makedirs(results_dir)
                                os.chdir(results_dir)
                                train_model(hidden_size, batch_size, depthT, depthG, lr, models_dir)     


                                round_end = time.time()

                                print '\n'
                                print "Finished training %s" %round_name
                                print 'Time to complete round: %s' %(round_end - round_start)
                                print 'Total elapsed time: %s' %(round_end - start)
                              
    for lr in params['lr']:
        for batch_size in params['batch_size']:
            for depthT in params['depthT']:
                for depthG in params['depthG']:
                    for hidden_size in params['hidden_size']:
                        round_name='lr_%s_bs_%s_depthT_%s_depthG_%s_hs_%s' %(lr, batch_size, depthT, depthG, hidden_size)
                        path = cwd + round_name + '/'
                        results_dir = path+'results/'
                        if (lr, batch_size, depthT, depthG, hidden_size) not in skip:
                            round_start = time.time()
                            print '\n'
                            print 'Starting %s' %round_name
                            print'\n'
                            models_dir = path+'newmodels'

                            if not os.path.isdir(path):
                                os.makedirs(path)
                            if not os.path.isdir(models_dir):
                                os.makedirs(models_dir)
                            if not os.path.isdir(results_dir):
                                os.makedirs(results_dir)
                            os.chdir(results_dir)
                            os.system("python %s/iclr19-graph2graph/scripts/val_script.py --decode %s --dir %s --num %s --n_decode %s \
                            --data_dir %s --iclr_dir %s --sim_delta %s --predictors %s --prop_targets %s > %sbest_of_round.txt" %(args.iclr_dir, args.decode, models_dir, str(epochs), n_decode, data_dir, args.iclr_dir, args.sim_delta, args.predictors, args.prop_targets, 
                                                             path) )
                            n_epoch = 0
                            best_acc = 0.0
                            div_epoch = 0
                            best_div = 0
                            with open('%sbest_of_round.txt' %path, 'r') as f:
                                for line in f:
                                    l = line.strip()
                                    if 'Epoch with best model' in l:
                                        n_epoch = l.split(': ')[1]
                                    elif 'Accuracy for best model' in l:    
                                        best_acc  = l.split(': ')[1]
                                    elif 'Epoch with most diverse model' in l:
                                        div_epoch = l.split(': ')[1]
                                    elif 'Diversity of that model' in l:
                                        best_div = l.split(': ')[1]

                            best_results.append((path, n_epoch, best_acc))
                            diversity_results.append((path, div_epoch, best_div))
                            round_end = time.time()

                            print '\n'
                            print 'Finished validating %s' %round_name
                            print 'Time to complete round: %s' %(round_end - round_start)
                            print 'Total elapsed time: %s' %(round_end - start)

                        else:
                            print 'Skipped %s' %round_name
                            os.chdir(results_dir) 
                            n_epoch = 0
                            best_acc = 0.0
                            div_epoch = 0
                            best_div = 0
                            with open('%sbest_of_round.txt' %path, 'r') as f:
                                for line in f:
                                    l = line.strip()
                                    if 'Epoch with best model' in l:
                                        n_epoch = l.split(': ')[1]
                                    elif 'Accuracy for best model' in l:    
                                        best_acc  = l.split(': ')[1]
                                    elif 'Epoch with most diverse model' in l:
                                        div_epoch = l.split(': ')[1]
                                    elif 'Diversity of that model' in l:
                                        best_div = l.split(': ')[1]

                            best_results.append((path, n_epoch, best_acc))
                            diversity_results.append((path, div_epoch, best_div))

    srt_acc = sorted(best_results, key=lambda x: x[2], reverse=True)
    srt_div = sorted(diversity_results, key=lambda x: x[2], reverse=True)

    print "Best Accuracy", srt_acc[0]
    print "Best Diversity", srt_div[0]