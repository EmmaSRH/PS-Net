import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random

# import os
# os.environ['CUDA_VISIBLE_DEVICE'] = 1,2,3


def main(config):
    cudnn.benchmark = True
    cudnn.enabled = True

    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    # epoch = random.choice([100,150,200,250])
    epoch = 600
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)
    if config.type == 'urisc':
        train_path = '../U-RISC-DATASET/patchs/2048/train/'
        valid_path = '../U-RISC-DATASET/patchs/2048/val/'
        test_path = '../U-RISC-DATASET/patchs/2048/test/'

    if config.type == 'isbi':
        train_path = '../ISBI2012/train/'
        valid_path = '../ISBI2012/val/'
        test_path = '../ISBI2012/test/'

    if config.type == 'road':
        train_path = '../Road/tiff/train/'
        valid_path = '../Road/tiff/val/'
        test_path = '../Road/tiff/test/'

    if config.type == 'cracktree':
        train_path = '../CrackTree/image/'
        valid_path = '../CrackTree/image/'
        test_path = '../CrackTree/image/'

    if config.type == "drive":
        train_path = '../DRIVE/training/images/'
        valid_path = '../DRIVE/test/images/'
        test_path = '../DRIVE/test/images/'
        
    train_loader = get_loader(image_path=train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            resize_size = config.resize_size,
                            mode='train',
                            augmentation_prob=config.augmentation_prob,
                            type=config.type)
    valid_loader = get_loader(image_path=valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            resize_size=config.resize_size,
                            mode='valid',
                            augmentation_prob=0.,
                            type=config.type)
    test_loader = get_loader(image_path=test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            resize_size=config.resize_size,
                            mode='test',
                            augmentation_prob=0.,
                            type=config.type)

    solver = Solver(config, train_loader, valid_loader, test_loader,[0,1,2,3,4,5,6,7])

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=1500)
    parser.add_argument('--resize_size', type=int, default=512)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=64)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--num_workers', type=int, default=96)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_name', type=str, default='U_Net-249.pkl', help='U_Net-249.pkl')
    parser.add_argument('--prediction_path', type=str, default='./predictions/')
    parser.add_argument('--model_path', type=str, default='./trained_models/')
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--type', type=str, default='drive')
    parser.add_argument('--cuda_idx', type=int, default=3)

    config = parser.parse_args()
    main(config)
