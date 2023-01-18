class Parameters:
    device = 'cpu'
    #  device = 'cuda'
    data_dir = r'./data'




    out_dim = 10  # 数字的类别有10种

    seed = 1234

    batch_size = 64  # 批大小
    init_lr = 1e-3  # 初始学习率
    epochs = 10  # 训练轮数
    verbose_step = 10  # 每10step验证一次
    save_step = 200  # 每200步保存一次


parametes = Parameters()