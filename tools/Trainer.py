import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
import time

class ModelNetTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, \
                 model_name, log_dir, num_views=12, n_classes=40, no_cuda=False):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views
        self.n_classes = n_classes
        self.no_cuda = no_cuda

        if self.no_cuda:
            self.model
        else:
            self.model.cuda()

        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)


    def train(self, n_epochs, max_batches=None):

        best_acc = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):
            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            if self.log_dir is not None:
                self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):
                if max_batches is not None and i >= max_batches:
                    print("max batches reached, skipping rest")
                    break
                if self.model_name == 'mvcnn':
                    N,V,C,H,W = data[1].size()
                    in_data = Variable(data[1]).view(-1,C,H,W)
                    if not self.no_cuda:
                        in_data = in_data.cuda()
                else:
                    in_data = Variable(data[1])
                    if not self.no_cuda:
                        in_data = in_data.cuda()
                target = Variable(data[0])
                if not self.no_cuda:
                    target = target.cuda()
                target = target.long()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)

                loss = self.loss_fn(out_data, target)
                
                if self.log_dir is not None:
                    self.writer.add_scalar('train/train_loss', loss, i_acc+i+1)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float()/results.size()[0]
                if self.log_dir is not None:
                    self.writer.add_scalar('train/train_overall_acc', acc, i_acc+i+1)

                loss.backward()
                self.optimizer.step()
                
                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch+1, i+1, loss, acc)
                if (i+1)%1==0:
                    print(log_str)
            i_acc += i

            # evaluation
            if (epoch+1)%1==0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch, max_batches)
                if self.log_dir is not None:
                    self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch+1)
                    self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch+1)
                    self.writer.add_scalar('val/val_loss', loss, epoch+1)

            # save best model
            if self.log_dir is not None and val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model.save(self.log_dir, epoch)
 
            # adjust learning rate manually
            if epoch > 0 and (epoch+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

        # export scalar data to JSON for external processing
        if self.log_dir is not None:
            self.writer.export_scalars_to_json(self.log_dir+"/all_scalars.json")
            self.writer.close()

    def update_validation_accuracy(self, epoch, max_batches=None):
        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        wrong_class = np.zeros(self.n_classes)
        samples_class = np.zeros(self.n_classes)
        all_loss = 0
        confusion = np.zeros((self.n_classes, self.n_classes))

        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred = []

        for batch, data in enumerate(self.val_loader, 0):
            if max_batches is not None and batch >= max_batches:
                print("max batches reached, skipping rest")
                break
            if self.model_name == 'mvcnn':
                N,V,C,H,W = data[1].size()
                in_data = Variable(data[1]).view(-1,C,H,W)
                if not self.no_cuda:
                    in_data = in_data.cuda()
            else:#'svcnn'
                in_data = Variable(data[1])
                if not self.no_cuda:
                    in_data = in_data.cuda()
            target = Variable(data[0])
            if not self.no_cuda:
                target = target.cuda()
                
            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

            # confusion matrix
            pred_local = pred.cpu().data.numpy().astype('int')
            target_local = target.cpu().data.numpy().astype('int')
            for i in range(results.size()[0]):
                pred_i = pred_local[i]
                target_i = target_local[i]
                confusion[pred_i][target_i] += 1

        print ('Total # of test models: ', all_points)
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print ('val mean class acc. : ', val_mean_class_acc)
        print ('val overall acc. : ', val_overall_acc)
        print ('val loss : ', loss)

        confusion /= np.sum(confusion)
        with np.printoptions(precision=3, suppress=True):
            print("confusion:")
            print(confusion)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc

