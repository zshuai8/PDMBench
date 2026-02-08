import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, \
    WPMixer, MultiPatchFormer, MLP, LSTM, SVM, XGB
import time


class DummyLogger:
    """Dummy logger when WandB is not available or disabled."""
    def log(self, *args, **kwargs):
        pass
    def init(self, *args, **kwargs):
        pass
    def finish(self, *args, **kwargs):
        pass


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'MLP': MLP,
            'LSTM': LSTM,
            'SVM': SVM,
            'XGB': XGB,
            'NT': Nonstationary_Transformer,  # Alias for Nonstationary_Transformer
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        if args.model == 'Chronos':
            print('Loading CHRONOS foundation model. Install with: pip install chronos-forecasting')
            from models import Chronos
            self.model_dict['Chronos'] = Chronos

        if args.model == 'Moirai':
            print('Loading MOIRAI foundation model. Install with: pip install uni2ts')
            from models import Moirai
            self.model_dict['Moirai'] = Moirai

        if args.model == 'LagLlama':
            print('Loading Lag-Llama foundation model. Install with: pip install lag-llama')
            from models import LagLlama
            self.model_dict['LagLlama'] = LagLlama

        if args.model == 'MOMENT':
            print('Loading MOMENT foundation model. Install with: pip install momentfm')
            from models import MOMENT
            self.model_dict['MOMENT'] = MOMENT

        if args.model == 'TimesFM':
            print('Loading TimesFM foundation model. Install with: pip install timesfm')
            from models import TimesFM
            self.model_dict['TimesFM'] = TimesFM

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        # Initialize logging (WandB optional)
        self.wandb = self._init_logging()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _init_logging(self):
        """Initialize logging - WandB if available and enabled, otherwise dummy logger."""
        use_wandb = getattr(self.args, 'use_wandb', False)
        if use_wandb:
            try:
                import wandb
                dataset_name = self.args.root_path.rstrip('/').split('/')[-1]
                wandb.init(
                    project=f"PDMBench-{dataset_name}",
                    name=f"{self.args.model}_{int(time.time())}",
                    config={
                        "model": self.args.model,
                        "learning_rate": self.args.learning_rate,
                        "epochs": self.args.train_epochs,
                        "batch_size": self.args.batch_size,
                        "seq_len": self.args.seq_len,
                    },
                )
                print("WandB logging enabled")
                return wandb
            except ImportError:
                print("WandB not installed. Using local logging.")
                return DummyLogger()
            except Exception as e:
                print(f"WandB initialization failed: {e}. Using local logging.")
                return DummyLogger()
        return DummyLogger()

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
