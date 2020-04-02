import os

params = {'lr': [.001, .005, .0002],
         'batch_size': [8, 32, 128],
          'depthT': [3, 6, 10],
          'depthG': [3, 5, 8]
         }

epochs = 5

n_decode = 20

bg_path = '~/CS6250_project/models/trial2/all_features/model.pkl'

best_results = []

cwd = os.getcwd() + '/'

for lr in params['lr']:
    for batch_size in params['batch_size']:
        for depthT in params['depthT']:
            for depthG in params['depthG']:
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
                os.system('python /home/rgur/CS6250_project/scripts/val_script.py %s %s %s %s %s > %sbest_of_round.txt' %(models_dir, str(epochs), n_decode, bg_path, data_dir, path) )
                with open('%sbest_of_round.txt' %path, 'r') as f:
                    for line in f:
                        l = line.strip()
                        if 'Epoch' in l:
                            n_epoch = l.split(': ')[1]
                        elif 'Accuracy' in l:    
                            best_acc  = l.split(': ')[1]
                best_results.append((path, n_epoch, best_acc))
                
srt = sorted(best_results, key=lambda x: x[2], reverse=True)
print srt[0]