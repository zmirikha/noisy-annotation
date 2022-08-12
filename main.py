import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import os
import torchvision.transforms as t
import matplotlib.pyplot as plt

from utils.dataset import Skin
from utils.avg import AverageMeter
from utils.evaluation import Evaluation
from models.unet_rw import *
from settings import get_arguments


def load_data(args):
    im_transform = t.Compose([t.ToTensor(), t.Normalize(args.mean, args.std)])

    dset = Skin(args.imgdir, args.gtdir, input_size=(args.img_width, args.img_height),
                im_transform=im_transform, target_transform=t.ToTensor())

    dataset_size = len(dset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size

    dsetTrain, dsetVal = random_split(dset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))

    train_data_loader = DataLoader(dsetTrain.dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)

    val_data_loader = DataLoader(dsetVal.dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

    dsetCl = Skin(args.gtcleandir, args.imgcleandir, im_transform=im_transform, target_transform=t.ToTensor())
    cl_data_loader = DataLoader(dsetCl, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers)
    return train_data_loader, val_data_loader, cl_data_loader


def train(args):
    model, opt, epoch = make_model(args)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=args.scheduler_milestone, gamma=args.scheduler_gamma)

    tr_losses = AverageMeter()

    tLoader, vLoader, clLoader = load_data(args)
    print('data loading done')

    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('starting epoch loop')
    for epoch in range(1, args.num_epochs + 1):

        tr_losses.reset()
        scheduler.step()

        cl_data_iterator = iter(clLoader)

        for step, (images, labels) in enumerate(tLoader):
            model.train(True)

            criterion = criterion.to(device)
            # Create the meta network
            meta_net = UNet_rw(1)
            meta_net.load_state_dict(model.state_dict())
            meta_net.to(device)

            images = to_var(images, requires_grad=False)
            labels = to_var(labels, requires_grad=False)

            y_f_hat = meta_net(images)

            cost = F.binary_cross_entropy_with_logits(y_f_hat.squeeze(), labels.squeeze(), reduce=False)

            # Computing weighted loss
            eps = to_var(torch.zeros(cost.size()))
            l_f_meta = torch.sum(cost * eps)

            meta_net.zero_grad()

            # Perform a parameter update
            grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
            meta_net.update_params(scheduler.get_last_lr()[0], source_params=grads)

            # 2nd forward pass and getting the gradients with respect to epsilon
            val_data = next(cl_data_iterator, None)
            if val_data is not None:
                val_im, val_gt = val_data
            else:
                cl_data_iterator = iter(clLoader)
                val_data = next(cl_data_iterator, None)
                val_im, val_gt = val_data

            val_im = to_var(val_im, requires_grad=False)
            val_gt = to_var(val_gt, requires_grad=False)

            y_g_hat = meta_net(val_im)
            l_g_meta = F.binary_cross_entropy_with_logits(y_g_hat, val_gt)

            grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

            # Compute and normalize the weights
            w_tilde = torch.clamp(-grad_eps, min=0)
            norm_c = torch.sum(w_tilde)

            if norm_c != 0:
                w = w_tilde / norm_c
            else:
                w = w_tilde

            # Compute the loss with the computed weights and then perform a parameter update
            y_f_hat = model(images)
            cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduce=False)

            l_f = torch.sum(cost.squeeze() * w)

            opt.zero_grad()
            l_f.backward()
            opt.step()

            tr_losses.update(l_f.data.cpu().numpy())

        vl_loss, vl_jacc, vl_dice, vl_spec, vl_sens, vl_accu = evaluate(model, vLoader, criterion=criterion)
        _, _, train_dice, _, _, train_accu = evaluate(model, tLoader)

        print('[Epoch: {0:02}/{1:02}]'
              '\t[TrainLoss: {2:.4f}]'
              '\t[TrDice: {3:.4f}]'
              '\t[TrAccu: {4:.4f}]'
              '\t[ValiLoss: {5:.4f}]'
              '\t[ValiJaccard: {6:.4f}]'
              '\t[ValiDice: {7:.4f}]'
              '\t[ValiSpec: {8:.4f}]'
              '\t[ValiSens: {9:.4f}]'
              '\t[ValiAccu: {10:.4f}]'.format(epoch, args.num_epochs, tr_losses.avg, train_dice, train_accu,
                                              vl_loss, vl_jacc,
                                              vl_dice, vl_spec, vl_sens, vl_accu))

        if epoch % args.log_step == 0:
            filename = f'checkpoint_{epoch:02}.pth.tar'
            state_dict = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': opt.state_dict()}
            fullname = os.path.join(args.savedir, filename)
            torch.save(state_dict, fullname)


def evaluate(model, val_loader, criterion=None):
    model.eval()

    losses = AverageMeter()
    jaccards = AverageMeter()
    dices = AverageMeter()
    specificity = AverageMeter()
    sensitivity = AverageMeter()
    accuracy = AverageMeter()

    eva = Evaluation()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            outputs = torch.sigmoid(logits)

            if criterion:
                criterion = criterion.to(device)
                loss = criterion(outputs, labels)
                losses.update(loss.cpu().numpy())

            jacc_index = eva.jaccard_similarity_coefficient(outputs.cpu().numpy(),
                                                            labels.cpu().numpy())
            dice_index = eva.dice_coefficient(outputs.cpu().numpy(),
                                              labels.cpu().numpy())

            specificity_index, sensitivity_index, accuracy_index = eva.specificity_sensitivity(labels.cpu().numpy(),
                                                                                               outputs.cpu().numpy())

            jaccards.update(jacc_index)
            dices.update(dice_index)
            specificity.update(specificity_index)
            sensitivity.update(sensitivity_index)
            accuracy.update(accuracy_index)

    return losses.avg, jaccards.avg, dices.avg, specificity.avg, sensitivity.avg, accuracy.avg


def make_model(args):
    model = UNet_rw(1)

    if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
        model = model.to('cuda')
        torch.backends.cudnn.benchmark = True

    opt = optim.SGD(model.params(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.99)

    epoch = 0
    if args.resume:
        filename = f'checkpoint_{args.state:02}.pth.tar'
        checkpoint = torch.load(os.path.join(args.savedir, filename))
        model.load_state_dict(checkpoint['state_dict'])
        opt.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

    return model, opt, epoch


def inference(args):
    im_transform = t.Compose([t.ToTensor(), t.Normalize(args.mean, args.std)])
    dsetInf = Skin(args.testimgdir, args.testgtdir, im_transform=im_transform,
                   target_transform=t.ToTensor())
    inf_data_loader = DataLoader(dsetInf, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filename = f'checkpoint_{args.state:02}.pth.tar'
    checkpoint = torch.load(os.path.join(args.modeldir, filename))

    model = UNet_rw(1)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    toImage = t.ToPILImage()

    with torch.no_grad():
        for step, (images, _) in enumerate(inf_data_loader):

            images = images.to(device)

            logits = model(images)
            outputs = torch.sigmoid(logits)

            filenames = inf_data_loader.dataset.img_filenames
            for j in range(outputs.size()[0]):
                index = step * args.batch_size + j
                output_img = toImage(outputs[j].cpu())
                res_filename = os.path.join(args.savedir, filenames[index].split('/')[-1])
                output_img.save(res_filename, 'BMP')

    _, jacc, dice, spec, sens, accu = evaluate(model, inf_data_loader)
    print(f'Inference Jaccard:{jacc:.2%}, Inference Dice:{dice:.2%}, Inference Specificity:{spec:.2%},'
          f'Inference Sensitivity:{sens:.2%}, Inference Accuracy:{accu:.2%}')


def main(args):
    if args.phase == 'train':
        train(args)
    elif args.phase == 'inference':
        inference(args)


if __name__ == '__main__':
    args = get_arguments()
    main(args)
