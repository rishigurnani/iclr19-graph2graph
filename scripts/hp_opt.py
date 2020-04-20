import os
import sys
from multiprocessing import Pool
import time

args = sys.argv
#n_core = args[1]

skip = [] #add hyperparameter combos which should be skipped

params = {'lr': [.001, .005, .0002],
         'batch_size': [8, 32],
          'depthT': [6],
          'depthG': [3, 8]
         }

# params = {'lr': [.0002],
#          'batch_size': [128],
#           'depthT': [6],
#           'depthG': [3]
#          }

epochs = 4 #default is 5

n_decode = 8 #default is 10

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
                    os.system('python /home/rgur/iclr19-graph2graph/diff_vae/vae_train.py --train %s --vocab %svocab.txt --save_dir %s \
                --hidden_size 300 --rand_size 16 --epoch %s --anneal_rate 0.8 --lr %s --batch_size %s --depthT %s --depthG %s | tee %s/LOG' %(processed_dir, data_dir, models_dir, epochs, lr, batch_size, depthT, depthG, models_dir) )
                    os.system('python /home/rgur/iclr19-graph2graph/scripts/val_script.py %s %s %s %s %s > %sbest_of_round.txt' %(models_dir, str(epochs), n_decode, bg_path, data_dir, path) )
                    
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
                    print 'Finished lr_%s_bs_%s_depthT_%s_depthG_%s' %(lr, batch_size, depthT, depthG)
                    print 'Time to complete round: %s' %(round_end - round_start)
                    print 'Total elapsed time: %s' %(round_end - start)
              
                else:
                    print 'Skipped lr_%s_bs_%s_depthT_%s_depthG_%s' % (lr, batch_size, depthT, depthG)
                    path = cwd + 'lr_%s_bs_%s_depthT_%s_depthG_%s/' %(lr, batch_size, depthT, depthG)
                    results_dir = path+'results/'
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