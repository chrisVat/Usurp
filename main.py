import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from subset_sampler import get_sampler, SAMPLER_TECHNIQUES
from tensorboardX import SummaryWriter
from tqdm import tqdm


class ExperimentRunner():
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.sample_technique = args.st
        self.sampler = None
        self.model = self.load_model()
        self.optimizer = self.load_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        self.single_example_loss = nn.CrossEntropyLoss(reduction="none")
        self.current_epoch = 0
        self.save_checkpoints = args.save_checkpoints
        self.total_epochs = args.epochs
        self.exp_name = f"{args.lr}_{args.momentum}_{args.weight_decay}_{args.train_batch}_{args.k}"
        self.writer = SummaryWriter("logs/" + self.exp_name + "/")
        self.train_loader, self.test_loader = self.load_data()


    def load_data(self):
        # load cifar10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        
        self.sampler = get_sampler(technique=self.sample_technique, dataset_len=len(trainset), subset_percentage = args.k)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.train_batch,
                                                  sampler=self.sampler, num_workers=2)  
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.test_batch, shuffle=False, num_workers=2)
        return trainloader, testloader


    def load_model(self):
        # get resnet18, change last layer depending on dataset (todo)
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, 10)
        model = model.to(self.device)
        return model

    def load_optimizer(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        return optimizer

    def save_checkpoint(self):
        chkptdir = os.path.join(os.getcwd(), "checkpoints/" + self.exp_name + "/")
        if not os.path.exists(chkptdir):
            os.makedirs(chkptdir)

        model_lst = [x for x in os.listdir(chkptdir) if x.endswith(".pth")]
        if len(model_lst) > 2:
            os.remove(os.path.join(chkptdir, model_lst[0]))
        
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.criterion,
        }, os.path.join(chkptdir, f"checkpoint_{self.current_epoch}.pth"))

    def load_checkpoint(self):
        chkptdir = os.path.join(os.getcwd(), "checkpoints/" + self.exp_name + "/")
        if not os.path.exists(chkptdir):
            os.makedirs(chkptdir)
        ckpts = os.listdir(chkptdir)
        if len(ckpts) > 0:
            ckpt_path = os.path.join(chkptdir, ckpts[-1])
            checkpoint = torch.load(ckpt_path)
            self.current_epoch = checkpoint["epoch"] + 1
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.criterion = checkpoint["loss"]
            print(f"Loaded checkpoint from {ckpt_path}")

    def train(self, epoch):
        self.model.train()
        train_loss = 0.0
        correct = 0.0
        total = 0
        losses = [] # list of losses for each example
        corrects = [] # boolean list of whether or not the model got the example correct

        with tqdm(enumerate(self.train_loader), total=len(self.train_loader)) as t:
            for batch_idx, (data, target) in t:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                losses += self.single_example_loss(output, target).tolist()
                pred = output.argmax(dim=1, keepdim=True)
                corrects += pred.eq(target.view_as(pred)).tolist()

                self.optimizer.step()
                train_loss += loss.item()
                t.set_description(f"Tr_E [{epoch}/{self.total_epochs}], Loss: {train_loss /(total):.6f}  AvgAcc: {100. * correct / total:.3f}% ")

        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.sampler.feedback({"losses": losses, "corrects": corrects, "batch_size": self.args.train_batch})
        
    
    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(enumerate(self.test_loader), total=len(self.test_loader)) as t:
                for batch_idx, (data, target) in t:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += self.criterion(output, target).item()
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    total += target.size(0)
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    t.set_description(f"Ts_E [{epoch}/{self.total_epochs}]  Loss: {test_loss / total:.6f}  AvgAcc: {100. * correct / total:.3f}% ")
        test_loss /= len(self.test_loader.dataset)
        self.writer.add_scalar('test_loss', test_loss, epoch)
        if self.save_checkpoints:
            self.save_checkpoint()
    

    def run(self):
        for epoch in range(self.current_epoch, self.total_epochs):
            self.train(self.current_epoch)
            self.test(self.current_epoch)
            self.current_epoch += 1
        self.writer.close()


# maintain list of loss / accuracy, if loss is higher or accuracy is worse move closer to that values distance, else move away / increase distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resnet Finetuner")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--st", default="mtds", type=str, help="Sampling Technique", choices=SAMPLER_TECHNIQUES)
    parser.add_argument("--gpu", default=0, type=int, help="gpu id")
    parser.add_argument("--epochs", default=2, type=int, help="epochs")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight_decay")
    parser.add_argument("--train_batch", default=128, type=int, help="train batch size")
    parser.add_argument("--test_batch", default=512, type=int, help="test batch size")
    parser.add_argument("--save_checkpoints", default=False, type=bool, help="save_checkpoints")
    parser.add_argument("--k", default=0.1, type=float, help="subset percentage")
    args = parser.parse_args()    
    # torch.cuda.set_device(args.gpu)
    
    experiment = ExperimentRunner(args)
    experiment.run()


