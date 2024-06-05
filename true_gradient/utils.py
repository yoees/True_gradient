from __future__ import print_function
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import yaml
import argparse
import datetime
import socket
from collections import Counter, defaultdict 

# for 'DVS128-Gesture' dataset_io
#import dataset_io.torchneuromorphic.dvs_gestures.create_hdf5 as create_hdf5
import dataset_io.torchneuromorphic.dvs_gestures.dvsgestures_dataloaders as dvsgestures_dataloaders


from networks import *

def load_data(dataset, num_workers, ds, dt, T_train, T_test, batch_size):
    path = os.getcwd() + '/dataset/'
    print(path)

    # Load Fashion-MNIST
    if dataset == 'F-MNIST' :
        data_path = path + 'Fashion_MNIST'
        train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

        test_set = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    # Load DvsGesture
    elif dataset == 'DVSGesture':
        # manually download from https://www.research.ibm.com/dvsgesture/ 
        # and place under dataset/DVS128-Gesture/
        # create_events_hdf5('./dataset/DVS128-gesture/DvsGesture',
        #'./dataset/DVS128-gesture/dvs128-gestures.hdf5')
        #root = path  + 'dvsgesture/dvs_gestures_build19.hdf5'       
        root = path + 'DVS128-Gesture/dvs128-gestures.hdf5'
        train_loader, test_loader= dvsgestures_dataloaders.create_dataloader(
        root= root,
        batch_size=batch_size,
        chunk_size_train = T_train,
        chunk_size_test = T_test,
        ds=ds,
        dt=dt*1000,
        num_workers=num_workers,
        sample_shuffle=True,
        time_shuffle=False,
        drop_last=True)

    return train_loader, test_loader


def make_model(dataset, thresh, thresh_n, tau_m, tau_s, frate, dt, input_shape, num_conv_layers, kernel_size, channels,
                padding, pool_size, stride, num_dense_layers, n_dense, type_grad, lens, expo, mant, type_net, expo_lim1, expo_lim2, grad_max):
    if type_net == 'srm_snn':
        return srm_snn(dataset, thresh, tau_m, tau_s, frate, dt, input_shape, 
                num_conv_layers, kernel_size, channels, padding, pool_size, stride, 
                num_dense_layers, n_dense, type_grad, lens, expo, mant, expo_lim1, expo_lim2, grad_max) 
    elif type_net == 'lif_snn':
        return lif_snn(dataset, thresh, tau_m, tau_s, frate, dt, input_shape, 
                num_conv_layers, kernel_size, channels, padding, pool_size, stride, 
                num_dense_layers, n_dense, type_grad, lens, expo, mant, expo_lim1, expo_lim2, grad_max)  
    
    
def load_checkpoint(directory, name):
    PATH = os.getcwd() +  '/true_gradient/logs/main_training/default/' + directory +'/'+ name
    checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint)
    # epoch = checkpoint['epoch']
    # acc = checkpoint['acc']
    # acc_hist = checkpoint['acc_hist']
    # loss_train_hist = checkpoint['loss_train_hist']
    # loss_test_hist = checkpoint['loss_test_hist']
    # spike_train_hist = checkpoint['spike_train_hist']
    # spike_test_hist = checkpoint['spike_test_hist']
    
    return checkpoint

def save_model(names, model, acc, epoch, acc_hist, train_loss_hist, test_loss_hist, spike_train_hist, spike_test_hist):
    state = {
        'net': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'acc_hist': acc_hist,
        'loss_train_hist': train_loss_hist,
        'loss_test_hist': test_loss_hist,
        'spike_train_hist': spike_train_hist,
        'spike_test_hist': spike_test_hist  
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    
    torch.save(state, './checkpoint/' + names)
    
    best_acc = 0  # best test accuracy
    best_acc = max(acc_hist)
    
    if acc == best_acc:
        torch.save(state, './checkpoint/' + names + '_best')

def parse_args(default_params_file = 'parameters/params.yml'):
    parser = argparse.ArgumentParser(description='True-gradient algorithm')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu or cuda)') 
    parser.add_argument('--resume_from', type=str, default=None, metavar='path_to_logdir', 
                        help='Path to a previously saved checkpoint') 
    parser.add_argument('--params_file', type=str, default=default_params_file, 
                        help='Path to parameters file to load. Ignored if resuming from checkpoint') 
    parser.add_argument('--no_save', dest='no_save', action='store_true', 
                        help='Set this flag if you don\'t want to save results') 
    parser.add_argument('--save_dir', type=str, default='default', help='Name of subdirectory to save results in') 
    parser.add_argument('--verbose', type=bool, default=False, help='print verbose outputs') 
    parser.add_argument('--seed', type=int, default=-1, help='CPU and GPU seed') 
    parser.add_argument('--no_train', dest='no_train', action='store_true', help='Train model (useful for resume)')
     
    parsed, unknown = parser.parse_known_args() 
 
    for arg in unknown: 
        if arg.startswith(("-", "--")): 
            #you can pass any arguments to add_argument 
            parser.add_argument(arg, type=str) 
 
    args=parser.parse_args() 
     
    if args.no_save: 
        print('!!!!WARNING!!!!\n\nRESULTS OF THIS TRAINING WILL NOT BE SAVED\n\n!!!!WARNING!!!!\n\n') 
 
    return args 
 
def prepare_experiment(name, args): 
    from tensorboardX import SummaryWriter 
    if args.resume_from is None: 
        params_file = args.params_file 
        if not args.no_save: 
            current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S') 
            log_dir = os.path.join(os.getcwd() + '/logs/{0}/'.format(name), 
                                   args.save_dir, 
                                   current_time) 
            checkpoint_dir = os.path.join(log_dir, 'checkpoints') 
            if not os.path.exists(log_dir): 
                os.makedirs(log_dir) 
            from shutil import copy2 
            copy2(params_file, os.path.join(log_dir, 'params.yml'))
            writer = SummaryWriter(log_dir=log_dir) 
            print('Saving results to {}'.format(log_dir)) 
    else: 
        log_dir = args.resume_from 
        checkpoint_dir = os.path.join(log_dir, 'checkpoints') 
        params_file = os.path.join(log_dir, 'params.yml') 
        if not args.no_save: 
            writer = SummaryWriter(log_dir=log_dir) 
        print('Resuming model from {}'.format(log_dir)) 
 
    with open(params_file, 'r') as f: 
        import yaml 
        params = yaml.safe_load(f)   
         
    if not 'batches_per_epoch' in params: 
        params['batches_per_epoch']=-1 
 
    if not args.no_save:  
        directories = {'log_dir':log_dir, 'checkpoint_dir': checkpoint_dir} 
    elif args.no_save and (args.resume_from is not None): 
        directories = {'log_dir':log_dir, 'checkpoint_dir': checkpoint_dir} 
        writer=None 
    else: 
        directories = {'log_dir':'', 'checkpoint_dir':''} 
        writer=None 
 
    if args.seed != -1: 
        print("setting seed {0}".format(args.seed)) 
        torch.manual_seed(args.seed) 
        np.random.seed(args.seed) 
    params = defaultdict(lambda: None,params)
    return params, writer, directories 
 
def load_model_from_checkpoint(checkpoint_dir, net, opt, n_checkpoint=-1, device='cuda'): 
    ''' 
    checkpoint_dir: string containing path to checkpoints, as stored by save_checkpoint 
    net: torch module with state_dict function 
    opt: torch optimizers 
    n_checkpoint: which checkpoint to use. number is not epoch but the order in the ordered list of checkpoint files 
    device: device to use (TODO: get it automatically from net) 
    ''' 
    starting_epoch = 0 
    checkpoint_list = os.listdir(checkpoint_dir) 
    if checkpoint_list: 
        checkpoint_list.sort() 
        last_checkpoint = checkpoint_list[n_checkpoint] 
        checkpoint = torch.load(os.path.join(checkpoint_dir, last_checkpoint), map_location=device) 
        net.load_state_dict(checkpoint['model_state_dict']) 
        opt.load_state_dicts(checkpoint['optimizer_state_dict']) 
        starting_epoch = checkpoint['epoch'] 
        print('Resuming from epoch {}'.format(starting_epoch)) 
    return starting_epoch

# def save_checkpoint(epoch, checkpoint_dir, net, opt): 
#     if not os.path.exists(checkpoint_dir): 
#         os.makedirs(checkpoint_dir) 
#     torch.save({ 
#         'epoch'               : epoch, 
#         'model_state_dict'    : net.state_dict(), 
#         'optimizer_state_dict': opt.state_dict(), 
#         }, os.path.join(checkpoint_dir, 'epoch{:05}.tar'.format(epoch))) 
