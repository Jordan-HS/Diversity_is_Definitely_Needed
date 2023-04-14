#########################################################################################################################
#   Code for Diversity is Definitely Needed: Improving Model-Agnostic Zero-shot Classification via Stable Diffusion     #
#   Jordan Shipard, Arnold Wiliem, Kien Nguyen Thanh, Wei Xiang, Clinton Fookes                                         #
#########################################################################################################################

import os
import torch
import wandb
from data import CIFAR10Generated, CIFAR10DataModule, CIFAR100Generated, CIFAR100DataModule, EuroSATDatamodule
from models import mobilenet_v3_small, ViT
import torchmetrics
from tqdm import tqdm
from torchvision.models import resnet50, resnet101, convnext_base, convnext_small
from torch.utils.data import DataLoader, ConcatDataset
import argparse
import subprocess
from torch.utils.data import ConcatDataset, DataLoader

parser = argparse.ArgumentParser()

parser.add_argument("--model", dest="model", type=str, default="Vit-B")
parser.add_argument("--epoch", dest="epoch", type=int, default=50)
parser.add_argument("--batch_size", dest="batch_size", type=int, default=64)
parser.add_argument("--syn_amount", dest="syn_amount", type=float, default=1.0)
parser.add_argument("--dataset", dest="dataset", type=str, required=True)
parser.add_argument("--img_size", dest="img_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--wd", type=float)
parser.add_argument("--model_path", type=str)
parser.add_argument("--wandb", dest="wandb", action="store_true", default=False)
parser.add_argument("--syn_data_location", dest="syn_data_location", type=str, default=None)
parser.add_argument("--real_data_location", dest="real_data_location", type=str, default=None)

args = parser.parse_args()

EPOCHS = args.epoch
LR = args.lr
BATCH_SIZE = args.batch_size

if args.model_path is None:
    name = f"{args.model} - {args.dataset} {args.syn_amount} syn pretraining"
    save_path = f"saved_models_v3/{args.model}/{args.dataset}/"
    save_name = f"{args.model}_{args.epoch}_pretrained_{args.syn_amount}_gen_data.pt"
else: 
    name = f"{args.model} - {args.dataset} {args.syn_amount} syn finetune"
    save_path = f"saved_models_v3/{args.model}/{args.dataset}/"
    save_name = f"{args.model}_{args.epoch}_finetune_{args.syn_amount}_gen_data.pt"

if not os.path.exists(save_path):
    os.makedirs(save_path)


# Setup the generated and real datasets
if "cifar10_" in args.dataset:
    gen_datamodule = CIFAR10Generated(batch_size=BATCH_SIZE, root_dir=os.path.join(args.syn_data_location, args.dataset))
    real_datamodule = CIFAR10DataModule(batch_size=BATCH_SIZE, root_dir=args.real_data_location)
elif "cifar100_" in args.dataset:
    gen_datamodule = CIFAR100Generated(batch_size=BATCH_SIZE, root_dir=os.path.join(args.syn_data_location, args.dataset))
    real_datamodule = CIFAR100DataModule(batch_size=BATCH_SIZE, root_dir=args.real_data_location)
elif "eurosat_" in args.dataset:
    gen_datamodule = EuroSATDatamodule(batch_size=BATCH_SIZE, root_dir=os.path.join(args.syn_data_location, args.dataset))
    real_datamodule = EuroSATDatamodule(batch_size=BATCH_SIZE, root_dir=args.real_data_location)

    
real_datamodule.setup("test", img_size=args.img_size)
real_test_dataloader = real_datamodule.test_dataloader()


if args.syn_amount < 1.0:
    gen_datamodule.setup(split_amount=args.syn_amount, img_size=args.img_size)
else:
    gen_datamodule.setup(stage = "fit", img_size=args.img_size)
    
train_dataloader = gen_datamodule.train_dataloader()


                ### MODELS ###
if args.model == "MBV3":
    model = mobilenet_v3_small(num_classes=gen_datamodule.n_classes, in_chans=3)
elif args.model == "Vit-B":
    model = ViT(image_size=args.img_size, patch_size=16, num_classes=gen_datamodule.n_classes, dim=768 , depth=12, heads=12, mlp_dim=768 *4)
elif args.model == "Vit-S":
    model = ViT(image_size=args.img_size, patch_size=16, num_classes=gen_datamodule.n_classes, dim=448 , depth=12, heads=7, mlp_dim=448 *3)
elif args.model == "RS50":
    model = resnet50(num_classes=gen_datamodule.n_classes)
elif args.model == "RS101":
    model = resnet101(num_classes=gen_datamodule.n_classes)
elif args.model == "convnext":
    model = convnext_base(num_classes=gen_datamodule.n_classes)
elif args.model == "convnext-s":
    model = convnext_small(num_classes=gen_datamodule.n_classes)

if args.model_path is not None:
    model.load_state_dict(torch.load(args.model_path))

model.cuda()


optim = torch.optim.AdamW(params=model.parameters(), lr=LR, weight_decay=args.wd)
criterion = torch.nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=EPOCHS)
    
train_metric = torchmetrics.Accuracy().cuda()
test_metric = torchmetrics.Accuracy().cuda()


logger = wandb.init(job_type="results", project="DDN", name=name,
                                                        config={
                                                        "total_epochs": EPOCHS,
                                                        "optimiser":type(optim).__name__,
                                                        "lr": LR,
                                                        "batch_size":BATCH_SIZE,
                                                        "model": args.model,
                                                        "syn_pretrain_amount": args.syn_amount,
                                                        "dataset": args.dataset,
                                                        "img_size": args.img_size,
                                                        "weight_decay": args.wd}) if args.wandb else None

best_acc = 0 

for epoch in range(EPOCHS):
    train_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    train_bar.set_description(f"Epoch: {epoch}")

    for idx, batch in train_bar:
        if type(batch) == dict:
            images, labels = batch.values()
            images = images.float()
        else:
            images, labels = batch
        
        
        images, labels = images.cuda(), labels.cuda()

        output = model(images)

        loss = criterion(output, labels)
        train_metric.update(output, labels)
        logger.log({"train_loss":loss,
                    "epoch":epoch}) if args.wandb else None

        optim.zero_grad()
        loss.backward()
        optim.step()

    scheduler.step()

    if args.wandb:
        logger.log({"train_acc": train_metric.compute(),
                    "epoch": epoch,
                    "lr": optim.param_groups[0]['lr']})
    else:
        print(f"Train acc: {train_metric.compute()}")

    train_metric.reset()

    torch.save(model.state_dict(), os.path.join(save_path, save_name))

    if epoch % 5 == 0:
        with torch.no_grad():
            for idx, batch in enumerate(real_test_dataloader):
                if type(batch) == dict:
                    images, labels = batch.values()
                    images = images.float()
                else:
                    images, labels = batch
                images, labels = images.cuda(), labels.cuda()

                output = model(images)

                loss = criterion(output, labels)
                acc = test_metric.update(output, labels)
                logger.log({"test_loss": loss,
                            "epoch" : epoch}) if args.wandb else None
            acc = test_metric.compute()

            if args.wandb:
                logger.log({"test_acc": acc,
                            "epoch": epoch})
            else:
                print(f"Test acc: {acc}")

            test_metric.reset()
            
            if acc > best_acc:
                torch.save(model.state_dict(), os.path.join(save_path, "best_"+save_name))
                best_acc = acc
            
            
with torch.no_grad():
    for idx, batch in enumerate(real_test_dataloader):
        if type(batch) == dict:
            images, labels = batch.values()
            images = images.float()
        else:
            images, labels = batch
        images, labels = images.cuda(), labels.cuda()

        output = model(images)

        loss = criterion(output, labels)
        acc = test_metric.update(output, labels)
        logger.log({"test_loss":loss,
                    "epoch":epoch}) if args.wandb else None

    if args.wandb:
        logger.log({"test_acc": test_metric.compute(),
                    "epoch": epoch})
    else:
        print(f"Test acc: {test_metric.compute()}")

    test_metric.reset()