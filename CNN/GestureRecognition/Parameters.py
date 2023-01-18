import torch

class Parameters:

    seed = 1234
    device = "cuda" if torch.cuda.is_available() else "cpu";
    data_root = './data'
    cls_mapper_path = './data/cls_mapper.json'
    train_data_root = './data/Marcel-Train'
    test_data_root = './data/Marcel-Test'
    metadata_train_path = './data/train_hand_gesture.txt'
    metadata_eval_path = './data/eval_hand_gesture.txt'
    metadata_test_path = './data/test_hand_gesture.txt'

    classes_num = 6
    data_channels = 3
    conv_kernel_size = 3
    fc_dropout_prob = 0.3

    batch_size = 2
    init_lr = 5e-4
    epochs = 100
    verbose_step = 250
    save_step = 500


parameters = Parameters()