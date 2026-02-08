from data_provider.data_loader import PDMloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'PDM': PDMloader
}


def data_provider(args, flag):
    """
    Data provider factory function.

    Args:
        args: Argument namespace with data configuration
        flag: 'TRAIN', 'VAL', or 'TEST'

    Returns:
        Tuple of (dataset, dataloader)
    """
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification' or args.task_name == 'early_failure':
        data_set = Data(
            args = args,
            root_path=args.root_path,
            file_list=args.file_list,
            flag=flag,
        )

        # prefetch_factor requires num_workers > 0
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle_flag,
            'num_workers': args.num_workers,
            'drop_last': drop_last,
        }
        if args.num_workers > 0:
            loader_kwargs['prefetch_factor'] = 2
        data_loader = DataLoader(data_set, **loader_kwargs)
        return data_set, data_loader
    else:
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
