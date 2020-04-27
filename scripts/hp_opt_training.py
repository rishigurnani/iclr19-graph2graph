import os
import sys
from multiprocessing import Pool
import time
import argparse

parser = argparse.ArgumentParser(description='specify test data size or percent')
parser.add_argument("--lr_grid", type=float,
                    help="list of learning rates to try out", nargs='+', default = [.001])
parser.add_argument("--bs_grid", type=int,
                    help="list of batch sizes to try out", nargs='+', default=[32])
parser.add_argument("--dT_grid", type=int,
                    help="list of depthT values to try out", nargs='+', default=[6])

parser.add_argument("--dG_grid", type=int,
                    help="list of depthT values to try out", nargs='+', default=[8])

parser.add_argument("--epochs", type=int,
                    help="number of epochs", default=2)

parser.add_argument("--iclr_dir", type=str,
                    help="directory containing repository", default='/home/rgur/g2g_new')

args = parser.parse_args()

skip = [] #add hyperparameter combos which should be skipped

params = {'lr': args.lr_grid,
         'batch_size': args.bs_grid,
          'depthT': args.dT_grid,
          'depthG': args.dG_grid
         }

print "param dict: %s" %params
# params = {'lr': [.0002],
#          'batch_size': [128],
#           'depthT': [6],
#           'depthG': [3]
#          }

epochs = args.epochs

#n_decode = args.n_decode #default is 10

bg_path = '~/CS6250_project/models/trial2/all_features/model.pkl' #path to property predictor

best_results = []
diversity_results = []

cwd = os.getcwd() + '/'

round_tup = []

start = time.time()

for lr in params['lr']:
    for batch_size in params['batch_size']:
        for depthT in params['depthT']:
            for depthG in params['depthG']:
                if (lr, batch_size, depthT, depthG) not in skip:
                    round_start = time.time()
                    print '\n'
                    print 'Starting lr_%s_bs_%s_depthT_%s_depthG_%s' %(lr, batch_size, depthT, depthG)
                    print'\n'
                    path = cwd + 'lr_%s_bs_%s_depthT_%s_depthG_%s/' %(lr, batch_size, depthT, depthG)
                    models_dir = path+'newmodels'
                    data_dir = cwd + 'data/'
                    processed_dir = cwd + 'processed/'
                    results_dir = path+'results/'

                    if not os.path.isdir(path):
                        os.makedirs(path)
                    if not os.path.isdir(models_dir):
                        os.makedirs(models_dir)
                    if not os.path.isdir(results_dir):
                        os.makedirs(results_dir)
                    os.chdir(results_dir)
                    os.system('python %s/iclr19-graph2graph/diff_vae/vae_train.py --train %s --vocab %svocab.txt --save_dir %s \
                --hidden_size 300 --rand_size 16 --epoch %s --anneal_rate 0.8 --lr %s --batch_size %s --depthT %s --depthG %s | tee %s/LOG' %(args.iclr_dir,processed_dir, data_dir, models_dir, epochs, lr, batch_size, depthT, depthG, models_dir) )
                    
#                     n_epoch = 0
#                     best_acc = 0.0
#                     div_epoch = 0
#                     best_div = 0
#                     with open('%sbest_of_round.txt' %path, 'r') as f:
#                         for line in f:
#                             l = line.strip()
#                             if 'Epoch with best model' in l:
#                                 n_epoch = l.split(': ')[1]
#                             elif 'Accuracy for best model' in l:    
#                                 best_acc  = l.split(': ')[1]
#                             elif 'Epoch with most diverse model' in l:
#                                 div_epoch = l.split(': ')[1]
#                             elif 'Diversity of that model' in l:
#                                 best_div = l.split(': ')[1]
                  
#                     best_results.append((path, n_epoch, best_acc))
#                     diversity_results.append((path, div_epoch, best_div))
                    round_end = time.time()
                    
                    print '\n'
                    print 'Finished lr_%s_bs_%s_depthT_%s_depthG_%s' %(lr, batch_size, depthT, depthG)
                    print 'Time to complete round: %s' %(round_end - round_start)
                    print 'Total elapsed time to this point: %s' %(round_end - start)
              
                else:
                    print 'Skipped lr_%s_bs_%s_depthT_%s_depthG_%s' % (lr, batch_size, depthT, depthG)
                    path = cwd + 'lr_%s_bs_%s_depthT_%s_depthG_%s/' %(lr, batch_size, depthT, depthG)
                    results_dir = path+'results/'
                    os.chdir(results_dir) 
#                     n_epoch = 0
#                     best_acc = 0.0
#                     div_epoch = 0
#                     best_div = 0
#                     with open('%sbest_of_round.txt' %path, 'r') as f:
#                         for line in f:
#                             l = line.strip()
#                             if 'Epoch with best model' in l:
#                                 n_epoch = l.split(': ')[1]
#                             elif 'Accuracy for best model' in l:    
#                                 best_acc  = l.split(': ')[1]
#                             elif 'Epoch with most diverse model' in l:
#                                 div_epoch = l.split(': ')[1]
#                             elif 'Diversity of that model' in l:
#                                 best_div = l.split(': ')[1]
                  
#                     best_results.append((path, n_epoch, best_acc))
#                     diversity_results.append((path, div_epoch, best_div))
                    
# srt_acc = sorted(best_results, key=lambda x: x[2], reverse=True)
# srt_div = sorted(diversity_results, key=lambda x: x[2], reverse=True)

# print "Best Accuracy", srt_acc[0]
# print "Best Diversity", srt_div[0]