from __future__ import print_function
import time
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
args = parse_args(os.getcwd() + '/parameters/params_true.yml')

starting_epoch = 0

params, writer, dirs = prepare_experiment(name=__file__.split('/')[-1].split('.')[0], args = args)
log_dir = dirs['log_dir']
checkpoint_dir = dirs['checkpoint_dir']
print(log_dir)
print(checkpoint_dir)

# Load data
train_loader, test_loader = load_data(params['dataset'],
                                      params['num_workers'],
                                      params['ds'], 
                                      params['dt'], 
                                      params['num_steps_train'], 
                                      params['num_steps_test'], 
                                      params['batch_size'])

if params['loss_function'] == 'mse':
    criterion = nn.MSELoss()
else:
    criterion = nn.CrossEntropyLoss().to(device)

# Create model
model = make_model(params['dataset'], params['thresh'], params['thresh_n'], params['tau_m'], params['tau_s'], params['frate'], params['dt'],
                    params['input_shape'], params['num_conv_layers'], params['kernel_size'], params['channels'],
                    params['padding'], params['pool_size'], params['stride'], params['num_dense_layers'], params['n_dense'], params['type_grad'],
                    params['lens'], params['expo'], params['mant'], 'lif_snn', params['expo_lim1'], params['expo_lim2'], params['grad_max']).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['lr_decay_interval'], gamma=params['lr_decay_rate'])

acc_hist = list([])
train_loss_hist = list([])
test_loss_hist = list([])
spike_train_hist = list([])
spike_test_hist = list([])

def train(dataset, model, train_loader, criterion, epoch, optimizer, scheduler):
    model.train()
    train_loss = 0
    count = 0
    with tqdm(total=len(train_loader)) as pbar:
        for i, (images, labels) in enumerate(train_loader):
            model.zero_grad()
            optimizer.zero_grad()        
            outputs = model(images.to(device), params['batch_size'], params['num_steps_train'], params['spike_no_limit'])
            # outputs: Time-average spike number

            if dataset == 'DVSGesture':
                if params['loss_function'] == 'mse':
                    labels = torch.unique(labels, dim=1).reshape(params['batch_size'],-1)
                else:
                    labels = torch.argmax(labels[:,0,:], axis=1)
            else:
                if params['loss_function'] == 'mse':
                    labels = nn.functional.one_hot(labels, num_classes=10)
                else:
                    pass        
            loss = criterion(outputs.cpu(), labels.float())
            train_loss += loss.item() / len(train_loader)        
            loss.backward()
            optimizer.step()
            # print('Iteration: ', count, ', loss: ', loss)
            count += 1

            pbar.update(1)
    
    # learning rate scheduling # do not use scheduler
    #scheduler.step()
    #optimizer.param_groups[0]["lr"] = np.clip(optimizer.param_groups[0]["lr"], params['min_lr'], params['learning_rate'])
        
    return train_loss

def test(dataset, model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            model.zero_grad()
            outputs = model(inputs.to(device), params['batch_size'], params['num_steps_test'], params['spike_no_limit'])                    
            # outputs: Time-average spike number
            
            if params['loss_function'] == 'mse':
                if dataset == 'DVSGesture':
                    targets = torch.unique(targets, dim=1).reshape(params['batch_size'],-1)                
                else:
                    targets = nn.functional.one_hot(targets, num_classes=11)
            else:
                if dataset == 'DVSGesture':
                    targets = torch.argmax(targets[:,0,:], axis=1)
                else:
                    pass

            loss = criterion(outputs.cpu(), targets)
            test_loss += loss.item() / len(test_loader)
            _, predicted = outputs.cpu().max(1)
            _, targets_idx = targets.max(1)
            # print(targets_idx)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets_idx).sum().item())
            acc = 100. * float(correct) / float(total)           
    
    return test_loss, acc

for epoch in range(params['num_epochs']):
    start_time = time.time()
    
    train_loss = train(params['dataset'], model, train_loader, criterion, epoch, optimizer, scheduler)
    test_loss, acc = test(params['dataset'],model, test_loader, criterion)
    
    # spike_train_hist.append(spike_map_train)
    # spike_test_hist.append(spike_map_test)
    acc_hist.append(acc)
    train_loss_hist.append(train_loss)
    test_loss_hist.append(test_loss)

    best_acc = max(acc_hist)
    best_epoch = acc_hist.index(best_acc)

    print('-------------------------------- Epoch {}----------------------------------'.format(epoch))
    print("Learning rate: {:.5f}".format(optimizer.param_groups[-1]['lr']))
    print("Epoch: {}/{}.. ".format(epoch, params['num_epochs']).ljust(14),
              "Train Loss: {:.5f}.. ".format(train_loss).ljust(20),
              "Test Loss: {:.5f}.. ".format(test_loss).ljust(19),
              "Test Accuracy: {:.5f}".format(acc))        
    print('Time elasped: %.2f' %(time.time()-start_time), '\n')
    print('Best test accuracy so far {:.5f} at Epoch {}'.format(max(acc_hist), best_epoch))
    print('\n')

    if ((epoch) % params['save_interval']) == 0 and epoch != 0:
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_hist,
            'loss_train_record': train_loss_hist,
            'loss_test_record': test_loss_hist,
            # 'spike_train_record': spike_train_hist,
            # 'spike_test_record': spike_test_hist 
        }
        print('Saving...')
        torch.save(state, checkpoint_dir)
    
    if acc == best_acc:
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,            
            'loss_train': train_loss,
            'loss_test': test_loss,
            # 'spike_train': spike_map_train,
            # 'spike_test': spike_map_test 
        }
        print('Saving best...')
        torch.save(state, checkpoint_dir+'.best')