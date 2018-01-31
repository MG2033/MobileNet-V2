import torch.optim
import torch.nn as nn
from torch.optim.rmsprop import RMSprop
from torch.autograd import Variable
import shutil
from utils import AverageTracker
from tqdm import tqdm
from tensorboardX import SummaryWriter


class Train:
    def __init__(self, model, trainloader, valloader, args):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.args = args
        self.start_epoch = 0
        self.best_top1 = 0.0

        # Loss function and Optimizer
        self.loss = None
        self.optimizer = None
        self.create_optimization()

        # Model Loading
        self.load_pretrained_model()
        self.load_checkpoint(self.args.resume_from)

        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=args.summary_dir)

    def train(self):
        for cur_epoch in range(self.start_epoch, self.args.num_epochs):

            # Initialize tqdm
            tqdm_batch = tqdm(self.trainloader,
                              desc="Epoch-" + str(cur_epoch) + "-")

            # Learning rate adjustment
            self.adjust_learning_rate(self.optimizer, cur_epoch)

            # Meters for tracking the average values
            loss, top1, top5 = AverageTracker(), AverageTracker(), AverageTracker()

            # Set the model to be in training mode (for dropout and batchnorm)
            self.model.train()

            for data, target in tqdm_batch:

                if self.args.cuda:
                    data, target = data.cuda(async=self.args.async_loading), target.cuda(async=self.args.async_loading)
                data_var, target_var = Variable(data), Variable(target)

                # Forward pass
                output = self.model(data_var)
                cur_loss = self.loss(output, target_var)

                # Optimization step
                self.optimizer.zero_grad()
                cur_loss.backward()
                self.optimizer.step()

                # Top-1 and Top-5 Accuracy Calculation
                cur_acc1, cur_acc5 = self.compute_accuracy(output.data, target, topk=(1, 5))
                loss.update(cur_loss.data[0])
                top1.update(cur_acc1[0])
                top5.update(cur_acc5[0])

            # Summary Writing
            self.summary_writer.add_scalar("epoch-loss", loss.avg, cur_epoch)
            self.summary_writer.add_scalar("epoch-top-1-acc", top1.avg, cur_epoch)
            self.summary_writer.add_scalar("epoch-top-5-acc", top5.avg, cur_epoch)

            # Print in console
            tqdm_batch.close()
            print("Epoch-" + str(cur_epoch) + " | " + "loss: " + str(loss.avg) + " - acc-top1: " + str(
                top1.avg)[:7] + "- acc-top5: " + str(top5.avg)[:7])

            # Evaluate on Validation Set
            if cur_epoch % self.args.test_every == 0 and self.valloader:
                self.test(self.valloader)

            # Checkpointing
            is_best = top1.avg > self.best_top1
            self.best_top1 = max(top1.avg, self.best_top1)
            self.save_checkpoint({
                'epoch': cur_epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_top1': self.best_top1,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)

    def test(self, testloader):
        loss, top1, top5 = AverageTracker(), AverageTracker(), AverageTracker()

        # Set the model to be in testing mode (for dropout and batchnorm)
        self.model.eval()

        for data, target in testloader:
            if self.args.cuda:
                data, target = data.cuda(async=self.args.async_loading), target.cuda(async=self.args.async_loading)
            data_var, target_var = Variable(data, volatile=True), Variable(target, volatile=True)

            # Forward pass
            output = self.model(data_var)
            cur_loss = self.loss(output, target_var)

            # Top-1 and Top-5 Accuracy Calculation
            cur_acc1, cur_acc5 = self.compute_accuracy(output.data, target, topk=(1, 5))
            loss.update(cur_loss.data[0])
            top1.update(cur_acc1[0])
            top5.update(cur_acc5[0])

        print("Test Results" + " | " + "loss: " + str(loss.avg) + " - acc-top1: " + str(
            top1.avg)[:7] + "- acc-top5: " + str(top5.avg)[:7])

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, self.args.checkpoint_dir + filename)
        if is_best:
            shutil.copyfile(self.args.checkpoint_dir + filename,
                            self.args.checkpoint_dir + 'model_best.pth.tar')

    def compute_accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, idx = output.topk(maxk, 1, True, True)
        idx = idx.t()
        correct = idx.eq(target.view(1, -1).expand_as(idx))

        acc_arr = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc_arr.append(correct_k.mul_(1.0 / batch_size))
        return acc_arr

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        learning_rate = self.args.learning_rate * (self.args.learning_rate_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    def create_optimization(self):
        self.loss = nn.CrossEntropyLoss()

        if self.args.cuda:
            self.loss.cuda()

        self.optimizer = RMSprop(self.model.parameters(), self.args.learning_rate,
                                 momentum=self.args.momentum,
                                 weight_decay=self.args.weight_decay)

    def load_pretrained_model(self):
        try:
            print("Loading ImageNet pretrained weights...")
            pretrained_dict = torch.load(self.args.pretrained_path)
            self.model.load_state_dict(pretrained_dict)
            print("ImageNet pretrained weights loaded successfully.\n")
        except:
            print("No ImageNet pretrained weights exist. Skipping...\n")

    def load_checkpoint(self, filename):
        filename = self.args.checkpoint_dir + filename
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.start_epoch = checkpoint['epoch']
            self.best_top1 = checkpoint['best_top1']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.args.checkpoint_dir, checkpoint['epoch']))
        except:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.checkpoint_dir))
