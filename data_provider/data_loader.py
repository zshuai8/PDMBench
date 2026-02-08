import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from data_provider.imputation import Imputer, impute_missing
from data_provider.resampling import Resampler, resample_to_length
from data_provider.normalization import ExtendedNormalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


class PDMloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
        imputation_method: Method for handling missing values
            Options: 'linear', 'spline', 'knn', 'mice', 'ffill', 'bfill', 'mean', 'median', 'zero'
        resampling_method: Method for resampling time series
            Options: 'linear', 'cubic', 'nearest', 'subsample', 'average', None (no resampling)
        target_length: Target sequence length for resampling (if None, uses original length)
        normalization_method: Method for normalization
            Options: 'standardization', 'minmax', 'robust', 'winsorized', 'per_channel',
                     'per_sample_std', 'per_sample_minmax', 'none'
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None,
                 imputation_method=None, resampling_method=None, target_length=None,
                 normalization_method=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag

        # Get preprocessing parameters from args if not explicitly provided
        self.imputation_method = imputation_method or getattr(args, 'imputation_method', 'linear')
        self.resampling_method = resampling_method or getattr(args, 'resampling_method', None)
        self.target_length = target_length or getattr(args, 'target_length', None)
        self.normalization_method = normalization_method or getattr(args, 'normalization_method', 'standardization')

        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # Apply preprocessing pipeline
        self.feature_df = self._apply_preprocessing(self.feature_df)
        print(len(self.all_IDs))

    def _apply_preprocessing(self, df):
        """
        Apply preprocessing pipeline: imputation -> resampling -> normalization.

        Args:
            df: Input dataframe

        Returns:
            Preprocessed dataframe
        """
        # Step 1: Imputation (handle missing values)
        if df.isna().any().any():
            if self.imputation_method != 'linear':
                # Use extended imputation methods
                imputer = Imputer(method=self.imputation_method)
                df = pd.DataFrame(
                    imputer.fit_transform(df.values),
                    columns=df.columns,
                    index=df.index
                )
            else:
                # Use original linear interpolation
                grp = df.groupby(by=df.index)
                df = grp.transform(interpolate_missing)

        # Step 2: Resampling (if specified)
        if self.resampling_method is not None and self.target_length is not None:
            df = self._resample_df(df)

        # Step 3: Normalization
        normalizer = Normalizer(norm_type=self.normalization_method)
        df = normalizer.normalize(df)

        return df

    def _resample_df(self, df):
        """
        Resample all samples in the dataframe to target length.

        Args:
            df: Input dataframe with multi-index (sample_id, time_step)

        Returns:
            Resampled dataframe
        """
        resampler = Resampler(
            target_length=self.target_length,
            method=self.resampling_method,
            anti_alias=True
        )

        resampled_dfs = []
        for sample_id in df.index.unique():
            sample_data = df.loc[sample_id].values
            resampled_data, _ = resampler.resample(sample_data)

            # Create new index for resampled data
            new_index = pd.Index([sample_id] * self.target_length)
            resampled_df = pd.DataFrame(
                resampled_data,
                columns=df.columns,
                index=new_index
            )
            resampled_dfs.append(resampled_df)

        # Update max_seq_len
        self.max_seq_len = self.target_length

        return pd.concat(resampled_dfs, axis=0)

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.npz')]
        if len(input_paths) == 0:
            pattern='*.csv'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        # df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
        #                                                      replace_missing_vals_with='NaN')
        loaded = np.load(filepath, allow_pickle=True)
        df_restored = pd.DataFrame(loaded['data'], columns=loaded['columns'])
        # df = pd.DataFrame(df_restored['features'].tolist())
        df = pd.DataFrame(df_restored['features'])
        labels = df_restored['label']

        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        # Extract RUL values if available (for early failure prediction)
        self.rul_df = None
        self.has_rul = False
        if 'rul_percentage' in df_restored.columns:
            # Dataset 07 has rul_percentage (0 to 1, where 1 = healthy, 0 = failure)
            rul_values = df_restored['rul_percentage'].values.astype(np.float32)
            self.rul_df = pd.DataFrame(rul_values, columns=['rul'])
            self.has_rul = True
        elif 'estimated_rul' in df_restored.columns:
            # Dataset 05 has estimated_rul (cycles until failure)
            rul_values = df_restored['estimated_rul'].values.astype(np.float32)
            self.rul_df = pd.DataFrame(rul_values, columns=['rul'])
            self.has_rul = True

        lengths = df.applymap(lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]
        # self.max_seq_len = int(df.shape[1])

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        # Return RUL if available (for early failure prediction)
        if self.has_rul and self.rul_df is not None:
            rul = self.rul_df.iloc[self.all_IDs[ind]].values
            return self.instance_norm(torch.from_numpy(batch_x)), \
                   torch.from_numpy(labels), \
                   torch.from_numpy(rul)

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
    
    
