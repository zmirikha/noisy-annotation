import argparse

# define a namespace for train mode
default_train = argparse.Namespace()
default_train.BATCH_SIZE = 6
default_train.CL_BATCH_SIZE = 30
default_train.IMG_WIDTH = 96
default_train.IMG_HEIGHT = 96
default_train.LEARNING_RATE = 0.0001
default_train.MOMENTUM = 0.9
default_train.WEIGHT_DECAY = 0.00005
default_train.LOG_STEP = 5
default_train.NUM_EPOCHS = 300
default_train.NUM_WORKERS = 2
default_train.SCHEDULER_MILESTONE = [100, 200]
default_train.SCHEDULER_GAMMA = 0.1
default_train.GT_DIR = './data/training/noisy_gt'
default_train.IMG_DIR = './data/training/noisy_img'
default_train.GT_CLEAN_DIR = './data/training/clean_gt'
default_train.IMG_CLEAN_DIR = './data/training/clean_img'
default_train.STATE = 10 # for resume
default_train.SAVE_DIR = './checkpoints'
default_train.MEAN = [0.70769376, 0.59134567, 0.54666793]
default_train.STD = [0.15497926, 0.16345601, 0.17852926]


# define a namespace for inference mode
default_inference = argparse.Namespace()
default_inference.BATCH_SIZE = 16
default_inference.IMG_DIR = './data/test_img'
default_inference.GT_DIR = './data/test_gt'
default_inference.NUM_WORKERS = 2
default_inference.STATE = 70
default_inference.SAVE_DIR = './results'
default_inference.MODEL_DIR = './checkpoints'
default_inference.MEAN = [0.70769376, 0.59134567, 0.54666793]
default_inference.STD = [0.15497926, 0.16345601, 0.17852926]


def get_arguments():
    parser = argparse.ArgumentParser(description='Learning from noisy annotations')
    subparsers = parser.add_subparsers(dest='phase', required=True)

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('-bsize', '--batch-size', type=int, default=default_train.BATCH_SIZE)
    parser_train.add_argument('-clbsize', '--cl-batch-size', type=int, default=default_train.CL_BATCH_SIZE)
    parser_train.add_argument('--img-width', type=int, default=default_train.IMG_WIDTH)
    parser_train.add_argument('--img-height', type=int, default=default_train.IMG_HEIGHT)
    parser_train.add_argument('-lr', '--learning-rate', type=float, default=default_train.LEARNING_RATE)
    parser_train.add_argument('-mom', '--momentum', type=float, default=default_train.MOMENTUM)
    parser_train.add_argument('--weight-decay', type=float, default=default_train.WEIGHT_DECAY)
    parser_train.add_argument('--log-step', type=int, default=default_train.LOG_STEP)
    parser_train.add_argument('-nepoch', '--num-epochs', type=int, default=default_train.NUM_EPOCHS)
    parser_train.add_argument('--num-workers', type=int, default=default_train.NUM_WORKERS)
    parser_train.add_argument('--scheduler_milestone', type=list, default=default_train.SCHEDULER_MILESTONE)
    parser_train.add_argument('--scheduler_gamma', type=float, default=default_train.SCHEDULER_GAMMA)
    parser_train.add_argument('--gtdir', type=str, default=default_train.GT_DIR)
    parser_train.add_argument('--imgdir', type=str, default=default_train.IMG_DIR)
    parser_train.add_argument('--gtcleandir', type=str, default=default_train.GT_CLEAN_DIR)
    parser_train.add_argument('--imgcleandir', type=str, default=default_train.IMG_CLEAN_DIR)
    parser_train.add_argument('--resume', action='store_true')
    parser_train.add_argument('--state', type=int, default=default_train.STATE)
    parser_train.add_argument('--savedir', type=str, default=default_train.SAVE_DIR)
    parser_train.add_argument('--mean', type=list, default=default_train.MEAN)
    parser_train.add_argument('--std', type=list, default=default_train.STD)

    parser_inf = subparsers.add_parser('inference')
    parser_inf.add_argument('-bsize', '--batch-size', type=int, default=default_inference.BATCH_SIZE)
    parser_inf.add_argument('--testimgdir', type=str, default=default_inference.IMG_DIR)
    parser_inf.add_argument('--testgtdir', type=str, default=default_inference.GT_DIR)
    parser_inf.add_argument('--num-workers', type=int, default=default_inference.NUM_WORKERS)
    parser_inf.add_argument('--state', type=int, default=default_inference.STATE)
    parser_inf.add_argument('--savedir', type=str, default=default_inference.SAVE_DIR)
    parser_inf.add_argument('--modeldir', type=str, default=default_inference.MODEL_DIR)
    parser_inf.add_argument('--mean', type=list, default=default_inference.MEAN)
    parser_inf.add_argument('--std', type=list, default=default_inference.STD)

    parsed_args = parser.parse_args()
    return parsed_args
