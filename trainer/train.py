print("START")
import os
print("1")
import sys
print("2")
import time
print("3")
import random
print("4")
import torch
print("5")
import torch.backends.cudnn as cudnn
print("6")
import torch.nn as nn
print("7")
import torch.nn.init as init
print("8")
import torch.optim as optim
print("9")
import torch.utils.data
print("10")
from torch.cuda.amp import autocast, GradScaler
print("11")
import numpy as np
print("12")

from utils import CTCLabelConverter, AttnLabelConverter, Averager
print("13")
from dataset import hierarchical_dataset, AlignCollate,Batch_Balanced_Dataset
print("14")
from model import Model
print("15")
from test import validation
print("16")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    print("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        #table.add_row([name, param])
        total_params+=param
        print(name, param)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def train(opt, show_number = 2, amp=False):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    #Create automatically saved_models folder if not exist
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.experiment_name}/log_dataset.txt', 'a', encoding="utf8")
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    #valid_loader = torch.utils.data.DataLoader(
    #    valid_dataset, batch_size=min(32, opt.batch_size),
    #    shuffle=True,  # 'True' to check training progress with validation function.
    #    num_workers=int(opt.workers), prefetch_factor=512,
    #    collate_fn=AlignCollate_valid, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=min(32, opt.batch_size),
        shuffle=True,
        num_workers=int(opt.workers),
        prefetch_factor=512 if int(opt.workers) > 0 else None,
        collate_fn=AlignCollate_valid,
        pin_memory=True
    )
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    if opt.saved_model != '':
        pretrained_dict = torch.load(opt.saved_model)
        if opt.new_prediction:
            model.Prediction = nn.Linear(model.SequenceModeling_output, len(pretrained_dict['module.Prediction.weight']))  
        
        model = torch.nn.DataParallel(model).to(device) 
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.load_state_dict(pretrained_dict)
        if opt.new_prediction:
            model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, opt.num_class)  
            for name, param in model.module.Prediction.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            model = model.to(device) 
    else:
        # weight initialization
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        model = torch.nn.DataParallel(model).to(device)
    
    model.train() 
    print("Model:")
    print(model)
    count_parameters(model)
    
    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # freeze some layers
    try:
        if opt.freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if opt.freeze_SequenceModeling:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except:
        pass
    
    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.optim=='adam':
        #optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizer = optim.Adam(filtered_parameters)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a', encoding="utf8") as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter

    scaler = GradScaler()
    t1= time.time()
        
    while(True):
        # train part
        optimizer.zero_grad(set_to_none=True)
        
        if amp:
            with autocast():
                image_tensors, labels = train_dataset.get_batch()
                image = image_tensors.to(device)
                text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
                batch_size = image.size(0)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text).log_softmax(2)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    preds = preds.permute(1, 0, 2)
                    torch.backends.cudnn.enabled = False
                    cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                    torch.backends.cudnn.enabled = True
                else:
                    preds = model(image, text[:, :-1])  # align with Attention.forward
                    target = text[:, 1:]  # without [GO] Symbol
                    cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            scaler.scale(cost).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            image_tensors, labels = train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)
            if 'CTC' in opt.Prediction:
                preds = model(image, text).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.permute(1, 0, 2)
                torch.backends.cudnn.enabled = False
                cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                torch.backends.cudnn.enabled = True
            else:
                preds = model(image, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) 
            optimizer.step()
        loss_avg.add(cost)
        
        # Print training loss and validation result (on valid dataset) per 10 iterations.
        if i % 10 == 0:
            print(f"iter {i} | loss {cost.item():.4f}")
            
        # validation part
        if (i % opt.valInterval == 0) and (i!=0):
            print('training time: ', time.time()-t1)
            t1=time.time()
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a', encoding="utf8") as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                    infer_time, length_of_data = validation(model, criterion, valid_loader, converter, opt, device)
                model.train()

                # training loss and validation loss
                loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                
                #show_number = min(show_number, len(labels))
                
                #start = random.randint(0,len(labels) - show_number )
                # Safer Validation Result Display
                show_number = min(show_number, len(labels))
                start = random.randint(0, max(0, len(labels) - show_number))
                    
                for gt, pred, confidence in zip(labels[start:start+show_number], preds[start:start+show_number], confidence_score[start:start+show_number]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
                print('validation time: ', time.time()-t1)
                t1=time.time()
        # save model per 1e+4 iter.
        if (i + 1) % 1e+4 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.experiment_name}/iter_{i+1}.pth')

        if i == opt.num_iter:
            print('end the training')
            sys.exit()
        i += 1

# Main Method
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # dataset paths
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--valid_data', required=True)

    # dataset selection
    parser.add_argument('--select_data', type=str, default='en')
    parser.add_argument('--batch_ratio', type=str, default='1')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0')
    parser.add_argument('--sensitive', action='store_true')

    # model architecture
    parser.add_argument('--Transformation', type=str, default='None')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM')
    parser.add_argument('--Prediction', type=str, default='CTC')
    parser.add_argument('--decode', type=str, default='greedy')

    # training settings
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_iter', type=int, default=200)
    parser.add_argument('--valInterval', type=int, default=20)
    parser.add_argument('--workers', type=int, default=0)

    # image settings
    parser.add_argument('--imgH', type=int, default=64)
    parser.add_argument('--imgW', type=int, default=300)
    parser.add_argument('--batch_max_length', type=int, default=25)

    # character set
    parser.add_argument('--character', type=str, required=True)

    # experiment name
    parser.add_argument('--experiment_name', type=str, default='debug_run')

    # misc options
    parser.add_argument('--PAD', action='store_true')
    parser.add_argument('--rgb', action='store_true')

    parser.add_argument('--contrast_adjust', type=float, default=0.0)

    # model dimensions
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--output_channel', type=int, default=512)
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--num_fiducial', type=int, default=20)

    # optimizer settings
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--rho', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--grad_clip', type=float, default=5)

    parser.add_argument('--saved_model', type=str, default='')

    parser.add_argument('--FT', action='store_true')
    parser.add_argument('--new_prediction', action='store_true')

    parser.add_argument('--freeze_FeatureFxtraction', action='store_true')
    parser.add_argument('--freeze_SequenceModeling', action='store_true')

    parser.add_argument('--data_filtering_off', action='store_true')

    parser.add_argument('--optim', type=str, default='adam')

    opt = parser.parse_args()

    print("MAIN METHOD RUNNING")
    print(opt)

    train(opt)