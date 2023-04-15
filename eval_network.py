import torch
from data import  CIFAR10DataModule, CIFAR100DataModule, EuroSATDatamodule
from models import mobilenet_v3_small, ViT
import torchmetrics
from torchvision.models import resnet50, resnet101, convnext_base, convnext_small
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model", dest="model", type=str, default="Vit-B")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=64)
parser.add_argument("--dataset", dest="dataset", type=str, required=True)
parser.add_argument("--img_size", dest="img_size", type=int, default=32)
parser.add_argument("--model_path", type=str)
parser.add_argument("--real_data_location", dest="real_data_location", type=str, default=None)

args = parser.parse_args()
BATCH_SIZE = args.batch_size

if args.dataset == "cifar10":
    real_datamodule = CIFAR10DataModule(batch_size=BATCH_SIZE, root_dir=args.real_data_location)
elif args.dataset == "cifar100":
    real_datamodule = CIFAR100DataModule(batch_size=BATCH_SIZE, root_dir=args.real_data_location)
elif args.dataset == "eurosat":
    real_datamodule = EuroSATDatamodule(batch_size=BATCH_SIZE, root_dir=args.real_data_location)

    
real_datamodule.setup("test", img_size=args.img_size)
real_test_dataloader = real_datamodule.test_dataloader()
test_metric = torchmetrics.Accuracy().cuda()


### MODEL
if args.model == "MBV3":
    model = mobilenet_v3_small(num_classes=real_datamodule.n_classes, in_chans=3)
elif args.model == "Vit-B":
    model = ViT(image_size=args.img_size, patch_size=16, num_classes=real_datamodule.n_classes, dim=768 , depth=12, heads=12, mlp_dim=768 *4)
elif args.model == "Vit-S":
    model = ViT(image_size=args.img_size, patch_size=16, num_classes=real_datamodule.n_classes, dim=448 , depth=12, heads=7, mlp_dim=448 *3)
elif args.model == "RS50":
    model = resnet50(num_classes=real_datamodule.n_classes)
elif args.model == "RS101":
    model = resnet101(num_classes=real_datamodule.n_classes)
elif args.model == "convnext":
    model = convnext_base(num_classes=real_datamodule.n_classes)
elif args.model == "convnext-s":
    model = convnext_small(num_classes=real_datamodule.n_classes)

if args.model_path is not None:
    model.load_state_dict(torch.load(args.model_path))

model.cuda()


with torch.no_grad():
    for idx, batch in enumerate(real_test_dataloader):
        if type(batch) == dict:
            images, labels = batch.values()
            images = images.float()
        else:
            images, labels = batch
        images, labels = images.cuda(), labels.cuda()

        output = model(images)

        acc = test_metric.update(output, labels)

    print(f"Test acc: {test_metric.compute()}")

    test_metric.reset()
