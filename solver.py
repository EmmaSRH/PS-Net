import os
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net,Light_UNet
import csv
from torchvision import transforms
from torchvision.transforms import ToTensor
from tensorboardX import SummaryWriter
import cv2
import time
from copy import deepcopy
from torchvision.models import resnet50
from losses import PHD_loss, focal_loss, dice_loss, sim_loss

torch.backends.cudnn.deterministic = True
from tqdm import tqdm
import torch.distributed as dist
import gc


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader, gpus):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.orig_size = (config.image_size, config.image_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bce_loss = torch.nn.BCELoss()
        self.phd_loss = PHD_loss.PerceptualHausdorfffLoss()
        self.focal_loss = focal_loss.FocalLossV3()
        self.dice_loss = dice_loss.DiceLoss()
        self.sim_loss = sim_loss.SimLoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.model_name = config.model_name
        self.result_path = config.result_path
        self.prediction_path = config.prediction_path
        self.mode = config.mode

        self.model_type = config.model_type
        self.t = config.t
        self.gpus = gpus
        self.build_model()

    def build_model(self):
        self.unet = U_Net(img_ch=3, output_ch=1)

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        # self.optimizer = torch.nn.DataParallel(self.optimizer, device_ids=self.gpus)

        self.unet.to(self.device)
        self.unet = torch.nn.DataParallel(self.unet.cuda(), device_ids=self.gpus, output_device=self.gpus[0])
        self.unet.cuda()

        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)
        return acc

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def load_pretrain_model(self):
        state_path = '/home/srh/Image_Segmentation/pretrained_models/cem500k_mocov2_resnet50_200ep_pth.tar'
        state = torch.load(state_path)
        print(list(state.keys()))
        # ['epoch', 'arch', 'state_dict', 'optimizer', 'norms']

        state_dict = state['state_dict']
        resnet50_state_dict = deepcopy(state_dict)
        for k in list(resnet50_state_dict.keys()):
            # only keep query encoder parameters; discard the fc projection head
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                resnet50_state_dict[k[len("module.encoder_q."):]] = resnet50_state_dict[k]

            # delete renamed or unused k
            del resnet50_state_dict[k]

        # as before we need to update parameter names to match the UNet model
        # for segmentation_models_pytorch we simply and the prefix "encoder."
        # format the parameter names to match torchvision resnet50
        unet_state_dict = deepcopy(resnet50_state_dict)
        for k in list(unet_state_dict.keys()):
            unet_state_dict['encoder.' + k] = unet_state_dict[k]
            del unet_state_dict[k]

        self.unet.load_state_dict(unet_state_dict, strict=False)

    def fusion(self, pre_global,pre_locals):

        return

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#
        ##### load pre-train from other data
        self.load_pretrain_model()

        ##### load pre-train from former epoch
        unet_path = ''
        writer = SummaryWriter(log_dir='U-Net-log')

        ##### Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            # Train for Encoder
            lr = self.lr
            # best_unet_score = 0

            for epoch in range(self.num_epochs):
                start = time.time()

                self.unet.train(True)
                epoch_loss = 0
                PHD = 0.
                length = 0

                for i, (filename, image, GT, img_crop_list, gt_crop_list) in enumerate(tqdm(self.train_loader)):

                    images = image.to(self.device)
                    GT = GT.to(self.device)
                    img_crop_list = img_crop_list.to(self.device)
                    gt_crop_list = gt_crop_list.to(self.device)

                    # ****************  global part training    ****************
                    pre = self.unet(images)
                    pre_global = torch.sigmoid(pre)
                    pre_flat = pre_global.view(pre_global.size(0), -1)
                    GT_flat = GT.view(GT.size(0), -1)
                    loss1 = self.bce_loss(pre_flat, GT_flat)
                    loss3 = self.focal_loss(pre_global, GT) # Focal loss
                    loss4 = self.dice_loss(pre_global, GT) # Dice loss
                    loss2 = self.phd_loss(pre_global, GT, tolerance=1)  # PHD loss
                    phd = loss2

                    if epoch < 5:
                        global_loss = loss1 + loss3 + loss4 # bce+focal+dice
                    else:

                        global_loss = loss1 + loss2 + loss3 + loss4

                    # ****************  local part training    ****************
                    pre_locals = [torch.sigmoid(self.unet(img_crop_list[:,i,:,:]))
                                  for i in range(4)]
                    local_loss = 0.
                    for i in range(len(pre_locals)):
                        pre_local = pre_locals[i]
                        pre_flat = pre_local.view(pre_global.size(0), -1)
                        GT_flat = GT.view(GT.size(0), -1)
                        loss1 = self.bce_loss(pre_flat, GT_flat)
                        loss3 = self.focal_loss(pre_local, GT)  # Focal loss
                        loss4 = self.dice_loss(pre_local, GT)  # Dice loss
                        if epoch < 5:
                            local_loss += loss1 + loss3 + loss4  # bce+focal+dice
                        else:
                            loss2 = self.phd_loss.forward(pre_global, GT, tolerance=1)  # PHD loss
                            local_loss += loss1 + loss2 + loss3 + loss4

                    # *************** sim part training ***************
                    sim_loss = self.sim_loss(pre_global,  pre_locals) # (N,1,512,512), ([(N,1,512,512)*4]]


                    epoch_loss = global_loss + local_loss + sim_loss
                    # Backprop + optimize
                    self.reset_grad()
                    epoch_loss.backward()
                    self.optimizer.step()

                    PHD += phd
                    length += 1

                    # write to web
                    writer.add_scalar('epoch_loss', np.mean(np.nan_to_num(epoch_loss.item())), epoch)
                    writer.add_image('image', images[0], epoch)
                    writer.add_image('gt', GT[0], epoch)
                    writer.add_image('prediction', 255 - pre_global[0] * 255., epoch)
                print('Training epoch {} times is {}: '.format(str(epoch), time.time() - start))
                # for each epoch, print a score
                PHD = PHD / length

                # Print the log info
                print(
                    'Epoch [%d/%d], Loss: %.4f, [Training] PHD: %.4f' % (epoch + 1, self.num_epochs, epoch_loss, PHD))

                # Decay learning rate
                if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Decay learning rate to lr: {}.'.format(lr))

                # ===================================== Validation ====================================#
                with torch.no_grad():
                    self.unet.train(False)
                    self.unet.eval()

                    PHD = 0. # PHD Score

                    length = 0
                    for i, (filename, images, GT, _, _) in enumerate(self.valid_loader):
                        images = images.to(self.device)
                        GT = GT.to(self.device)
                        SR = torch.sigmoid(self.unet(images))
                        PHD += self.phd_loss.forward(SR, GT, tolerance=1)

                        length += 1
                    # validation scores
                    PHD = PHD / length


                    print('[Validation] PHD: %.4f' % (PHD))

                    # # Save Best U-Net model
                    # if unet_score > best_unet_score:
                    #     best_unet_score = unet_score
                    #     best_epoch = epoch
                    #     best_unet = self.unet.state_dict()
                    #     print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
                    #     unet_path = os.path.join(self.model_path, 'best-f1-%s-%d.pkl' % (self.model_type, epoch))
                    #     torch.save(best_unet,unet_path)

                    unet_path = os.path.join(self.model_path, '%s-%d.pkl' % (self.model_type, epoch))
                    new_unet = self.unet.state_dict()
                    torch.save(new_unet, unet_path)

                    f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
                    wr = csv.writer(f)
                    wr.writerow(
                        [self.model_type,self.lr, self.num_epochs, self.num_epochs_decay,self.augmentation_prob])
                    f.close()

                torch.cuda.empty_cache()
                gc.collect()

