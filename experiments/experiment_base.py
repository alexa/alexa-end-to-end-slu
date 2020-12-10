# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import torch
import numpy as np
from utils.utils import AverageMeter
from tqdm import tqdm
from utils.visualize import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


class ExperimentRunnerBase:
    def __init__(self, args):
        # Set the LR Scheduler and Loss Parameters
        if args.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        factor=0.5,
                                                                        patience=3,
                                                                        mode='max',
                                                                        verbose=True)
        elif args.scheduler == 'cycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                                 max_lr=args.learning_rate,
                                                                 steps_per_epoch=len(self.train_loader),
                                                                 epochs=args.num_epochs)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.visualize = args.visualize
        if self.visualize:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter()

        # Training specific params
        self.args = args
        self.num_epochs = args.num_epochs
        self.print_every = args.print_every
        self.val_every = args.val_every
        self.model_dir = args.model_dir
        self.save_every = args.save_every

    def train(self):
        # Setting the variables before starting the training
        avg_train_loss = AverageMeter()
        avg_train_acc = AverageMeter()
        best_val_acc = -np.inf

        for epoch in range(self.num_epochs):
            avg_train_loss.reset()
            avg_train_acc.reset()

            # Mini batch loop
            for batch_idx, batch in enumerate(tqdm(self.train_loader)):
                step = epoch * len(self.train_loader) + batch_idx

                # Get the model output for the batch and update the loss and accuracy meters
                train_loss, train_acc = self.train_step(batch)
                if self.args.scheduler == 'cycle':
                    self.scheduler.step()
                avg_train_loss.update([train_loss.item()])
                avg_train_acc.update([train_acc])

                # Save the step checkpoint if needed
                if step % self.save_every == 0:
                    step_chkpt_path = os.path.join(self.model_dir,
                                                   'step_chkpt_{}_{}.pth'.format(epoch, step))
                    print("Saving the model checkpoint for epoch {} at step {}".format(epoch, step))
                    torch.save(self.model.state_dict(), step_chkpt_path)

                # Logging and validation check
                if step % self.print_every == 0:
                    print('Epoch {}, batch {}, step {}, '
                          'loss = {:.4f}, acc = {:.4f}, '
                          'running averages: loss = {:.4f}, acc = {:.4f}'.format(epoch,
                                                                                 batch_idx,
                                                                                 step,
                                                                                 train_loss.item(),
                                                                                 train_acc,
                                                                                 avg_train_loss.get(),
                                                                                 avg_train_acc.get()))

                if step % self.val_every == 0:
                    val_loss, val_acc = self.val()
                    print('Val acc = {:.4f}, Val loss = {:.4f}'.format(val_acc, val_loss))
                    if self.visualize:
                        self.writer.add_scalar('Val/loss', val_loss, step)
                        self.writer.add_scalar('Val/acc', val_acc, step)

                    # Update the save the best validation checkpoint if needed
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_chkpt_path = os.path.join(self.model_dir,
                                                       'best_ckpt.pth')
                        torch.save(self.model.state_dict(), best_chkpt_path)
                    if self.args.scheduler == 'plateau':
                        self.scheduler.step(val_acc)

                if self.visualize:
                    # Log data to
                    self.writer.add_scalar('Train/loss', train_loss.item(), step)
                    self.writer.add_scalar('Train/acc', train_acc, step)

    def compute_loss(self, batch):
        """ This function is specific to the kind of model we are training and must be implemented """
        raise NotImplementedError

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        metrics = self.compute_loss(batch)
        metrics['loss'].backward()
        self.optimizer.step()
        return metrics['loss'], metrics['accuracy']

    def load_model_for_eval(self):
        chkpt_path = os.path.join(self.model_dir, 'best_ckpt.pth') \
            if self.args.eval_checkpoint_path is None else self.args.eval_checkpoint_path
        self.model.load_state_dict(torch.load(chkpt_path))
        self.model.eval()

    @torch.no_grad()
    def val(self):
        print('VALIDATING:')
        avg_val_loss = AverageMeter()
        avg_val_acc = AverageMeter()

        self.model.eval()
        for batch_idx, batch in enumerate(tqdm(self.val_loader)):
            metrics = self.compute_loss(batch)
            avg_val_acc.update(metrics['correct'].cpu().numpy())
            avg_val_loss.update([metrics['loss']])
        return avg_val_loss.get(), avg_val_acc.get()

    @torch.no_grad()
    def infer(self):
        self.load_model_for_eval()
        avg_test_loss = AverageMeter()
        avg_test_acc = AverageMeter()
        all_true_labels = []
        all_pred_labels = []
        all_audio_embeddings = []
        all_text_embeddings = []

        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            # Get the model output and update the meters
            output = self.compute_loss(batch)
            avg_test_acc.update(output['correct'].cpu().numpy())
            avg_test_loss.update([output['loss']])

            # Store the Predictions
            all_true_labels.append(batch['label'].cpu())
            all_pred_labels.append(output['predicted'].cpu())
            all_audio_embeddings.append(output['model_output']['audio_embed'].cpu())
            all_text_embeddings.append(output['model_output']['text_embed'].cpu())

        # Collect the predictions and embeddings for the full set
        all_true_labels = torch.cat(all_true_labels).numpy()
        all_pred_labels = torch.cat(all_pred_labels).numpy()
        all_audio_embeddings = torch.cat(all_audio_embeddings).numpy()
        all_text_embeddings = torch.cat(all_text_embeddings).numpy()

        # Save the embeddings and plot the confusion matrix
        np.savez_compressed('embeddings.npz',
                            audio=all_audio_embeddings,
                            text=all_text_embeddings,
                            labels=all_true_labels)
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        plot_confusion_matrix(cm, self.test_loader.dataset.labels_list(), normalize=True)

        print('Final test acc = {:.4f}, test loss = {:.4f}'.format(avg_test_acc.get(), avg_test_loss.get()))
        return avg_test_loss.get(), avg_test_acc.get()
