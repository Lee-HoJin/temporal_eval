import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import acf
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from typing import Dict, List, Optional

import sys
from scipy.stats import ks_2samp

from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel

import warnings
warnings.filterwarnings('ignore')

import utils
from temporal_metrics.dependencies import ( transition_matrix_analysis,
                                            lag_k_diff_analysis,
                                            temporal_acf_comparison_2 )
from temporal_metrics.validity import ( evaluate_temporal_validity )

class TemporalBenchmark:
    """
    시계열 데이터를 위한 벤치마크 프레임워크
    - Automatic change point detection
    - Time binning 기반 cross-sectional analysis
    - Concept drift detection
    """
    
    def __init__(
        self,
        metadata,
        time_column: str = 'Date',
        bin_strategy: str = 'monthly',
        custom_bins: Optional[List] = None,
    ):
        self.metadata = metadata
        self.time_column = time_column
        self.bin_strategy = bin_strategy
        self.custom_bins = custom_bins
    
    def get_numeric_columns(self, df: pd.DataFrame, features: List[str], table_name) -> List[str]:
        """수치형 컬럼 추출 (dtype 기반)"""
        if self.metadata is None:
            # 메타데이터 없으면 기존 방식 사용
            return self.get_numeric_columns(df, features)
        
        numeric_cols = []
        table_meta = self.metadata['tables'][table_name]['columns']
        
        for col in features:
            if col not in df.columns or col not in table_meta:
                continue
            
            sdtype = table_meta[col].get('sdtype')

            if sdtype in ['numerical']:
                numeric_cols.append(col)
        
        return numeric_cols

    def get_categorical_columns(self, df: pd.DataFrame, features: List[str], table_name) -> List[str]:
        """범주형 컬럼 추출 (dtype 기반)"""
        if self.metadata is None:
            return self.get_categorical_columns(df, features)
        
        categorical_cols = []
        table_meta = self.metadata['tables'][table_name]['columns']
        
        for col in features:
            if col not in df.columns or col not in table_meta:
                continue
            
            sdtype = table_meta[col].get('sdtype')
            
            # categorical, boolean만 선택
            if sdtype in ['categorical', 'boolean']:
                categorical_cols.append(col)
        
        return categorical_cols
        
    def create_time_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간 구간별로 binning"""
        df = df.copy()
        
        # Date 컬럼이 string이면 datetime으로 변환
        if not pd.api.types.is_datetime64_any_dtype(df[self.time_column]):
            df[self.time_column] = pd.to_datetime(df[self.time_column])
        
        if self.bin_strategy == 'weekly':
            df['time_bin'] = df[self.time_column].dt.to_period('W').astype(str)
        elif self.bin_strategy == 'monthly':
            df['time_bin'] = df[self.time_column].dt.to_period('M').astype(str)
        elif self.bin_strategy == 'daily':
            df['time_bin'] = df[self.time_column].dt.to_period('D').astype(str)
        elif self.bin_strategy == 'quarterly':
            df['time_bin'] = df[self.time_column].dt.to_period('Q').astype(str)
        else:
            raise ValueError(f"Unknown bin strategy: {self.bin_strategy}")
                
        return df
        
    def bin_length_discrepancy(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        features: List[str],
    ) -> Dict:
        """
        시간 bin별 length discrepancy 계산
        """
        
        real_binned = self.create_time_bins(real_df)
        syn_binned = self.create_time_bins(syn_df)

        # for feat in features:
        #     print(f"Feature: {feat}")
        #     real_bin_counts = real_binned.groupby('time_bin')[feat].size().sort_index()
        #     print("Bin Counts (Real):")
        #     print(real_bin_counts)
            
        #     syn_bin_counts = syn_binned.groupby('time_bin')[feat].size().sort_index()
        #     print("Bin Counts (Synthetic):")
        #     print(syn_bin_counts)
        #     print()

        results = {
            "per_feature": {},
            "bin_length_discrepancy_overall": np.nan
        }

        flds = []

        for feat in features:
            if feat not in real_binned.columns:
                continue

            real_counts = (
                real_binned
                .groupby("time_bin")[feat]
                .count()
            )
            syn_counts = (
                syn_binned
                .groupby("time_bin")[feat]
                .count()
            )

            aligned = pd.concat(
                [real_counts, syn_counts],
                axis=1,
                keys=["real", "syn"]
            ).dropna()

            if aligned.empty:
                continue

            fld = np.mean(np.abs(aligned["real"] - aligned["syn"]) / (aligned["real"] + 1e-12))
            results["per_feature"][feat] = fld
            flds.append(fld)

        if flds:
            results["bin_length_discrepancy_overall"] = float(np.mean(flds))

        return results
    
    def temporal_mean_mae(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        features: List[str],
    ) -> Dict:
        """
        시간 bin별 평균의 MAE
        """
        real_binned = self.create_time_bins(real_df)
        syn_binned = self.create_time_bins(syn_df)
        
        results = {
            "per_feature": {},
            "temporal_mean_mae_overall": np.nan
        }

        maes = []

        for feat in features:
            if feat not in real_binned.columns:
                continue

            real_mean = (
                real_binned
                .groupby("time_bin")[feat]
                .mean()
            )
            syn_mean = (
                syn_binned
                .groupby("time_bin")[feat]
                .mean()
            )

            # 공통 bin만 비교
            aligned = pd.concat(
                [real_mean, syn_mean],
                axis=1,
                keys=["real", "syn"]
            ).dropna()

            if aligned.empty:
                continue

            mae = np.mean(np.abs(aligned["real"] - aligned["syn"]))
            results["per_feature"][feat] = mae
            maes.append(mae)

        if maes:
            results["temporal_mean_mae_overall"] = float(np.mean(maes))

        return results

    def temporal_var_mae(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        features: List[str],
        ddof: int = 1
    ) -> Dict:
        """
        시간 bin별 분산의 MAE
        """
        real_binned = self.create_time_bins(real_df)
        syn_binned = self.create_time_bins(syn_df)

        results = {
            "per_feature": {},
            "temporal_var_mae_overall": np.nan
        }

        maes = []

        for feat in features:
            if feat not in real_binned.columns:
                continue

            real_var = (
                real_binned
                .groupby("time_bin")[feat]
                .var(ddof=ddof)
            )
            syn_var = (
                syn_binned
                .groupby("time_bin")[feat]
                .var(ddof=ddof)
            )

            aligned = pd.concat(
                [real_var, syn_var],
                axis=1,
                keys=["real", "syn"]
            ).dropna()

            if aligned.empty:
                continue

            mae = np.mean(np.abs(aligned["real"] - aligned["syn"]))
            results["per_feature"][feat] = mae
            maes.append(mae)

        if maes:
            results["temporal_var_mae_overall"] = float(np.mean(maes))

        return results
    
    # TemporalBenchmark 클래스에 추가할 메서드들

   
    def temporal_jsd(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        features: List[str],
        min_count_per_bin: int = 5,
        base: float = 2.0,
    ) -> Dict:
        """
        for categorical features - uses frequency distribution
        """
        real_binned = self.create_time_bins(real_df)
        syn_binned = self.create_time_bins(syn_df)
        
        results = {"per_feature": {}, "temporal_jsd_overall": np.nan}
        scores = []

        for feat in features:
            if feat not in real_binned.columns or feat not in syn_binned.columns:
                continue

            common_bins = sorted(set(real_binned["time_bin"]).intersection(set(syn_binned["time_bin"])))
            if not common_bins:
                continue

            jsds = []
            for tb in common_bins:
                r = real_binned.loc[real_binned["time_bin"] == tb, feat].dropna()
                s = syn_binned.loc[syn_binned["time_bin"] == tb, feat].dropna()

                if len(r) < min_count_per_bin or len(s) < min_count_per_bin:
                    continue

                # 범주형 데이터: 빈도 분포로 JSD 계산
                f_real, f_syn = utils.get_frequencies(r.values, s.values)
                
                jsd = float(jensenshannon(f_real, f_syn, base=base))
                jsds.append(jsd)

            if jsds:
                feat_score = float(np.mean(jsds))
                results["per_feature"][feat] = feat_score
                scores.append(feat_score)

        if scores:
            results["temporal_jsd_overall"] = float(np.mean(scores))

        return results

    
    def temporal_wasserstein(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        features: List[str],
        min_count_per_bin: int = 5,
    ) -> Dict:
        """
        for numerical features
        """
        real_binned = self.create_time_bins(real_df)
        syn_binned = self.create_time_bins(syn_df)

        results = {"per_feature": {}, "temporal_wasserstein_overall": np.nan}
        scores = []

        for feat in features:
            if feat not in real_binned.columns or feat not in syn_binned.columns:
                continue

            real_binned[feat] = pd.to_numeric(real_binned[feat], errors="coerce")
            syn_binned[feat] = pd.to_numeric(syn_binned[feat], errors="coerce")

            common_bins = sorted(set(real_binned["time_bin"]).intersection(set(syn_binned["time_bin"])))
            if not common_bins:
                continue

            wds = []
            for tb in common_bins:
                r = real_binned.loc[real_binned["time_bin"] == tb, feat].dropna().values
                s = syn_binned.loc[syn_binned["time_bin"] == tb, feat].dropna().values

                if len(r) < min_count_per_bin or len(s) < min_count_per_bin:
                    continue

                wd = float(wasserstein_distance(r, s))
                wds.append(wd)

            if wds:
                feat_score = float(np.mean(wds))
                results["per_feature"][feat] = feat_score
                scores.append(feat_score)

        if scores:
            results["temporal_wasserstein_overall"] = float(np.mean(scores))

        return results
    
    def temporal_ks(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        features: List[str],
        min_count_per_bin: int = 5,
    ) -> Dict:
        """
        시간 bin별 feature의 KS-Complement 계산
        - KS statistic: 두 분포의 최대 차이 (0에 가까울수록 유사)
        - KS complement: 1 - KS statistic (1에 가까울수록 유사)
        
        for numerical features
        """
        real_binned = self.create_time_bins(real_df)
        syn_binned = self.create_time_bins(syn_df)

        results = {"per_feature": {}, "temporal_ks_overall": np.nan}
        scores = []
        
        MAX_DECIMALS = sys.float_info.dig - 1

        for feat in features:
            if feat not in real_binned.columns or feat not in syn_binned.columns:
                continue

            real_binned[feat] = pd.to_numeric(real_binned[feat], errors="coerce")
            syn_binned[feat] = pd.to_numeric(syn_binned[feat], errors="coerce")

            common_bins = sorted(set(real_binned["time_bin"]).intersection(set(syn_binned["time_bin"])))
            if not common_bins:
                continue

            ks_stats = []
            for tb in common_bins:
                r = real_binned.loc[real_binned["time_bin"] == tb, feat].dropna().values
                s = syn_binned.loc[syn_binned["time_bin"] == tb, feat].dropna().values

                if len(r) < min_count_per_bin or len(s) < min_count_per_bin:
                    continue

                # Round to avoid floating point precision issues
                r = r.round(MAX_DECIMALS)
                s = s.round(MAX_DECIMALS)

                try:
                    ks, _ = ks_2samp(r, s)
                    ks_stats.append(ks)
                except ValueError as e:
                    if 'must not be empty' in str(e):
                        continue
                    else:
                        raise

            if ks_stats:
                feat_score = float(np.mean(ks_stats))
                results["per_feature"][feat] = feat_score
                scores.append(feat_score)

        if scores:
            results["temporal_ks_overall"] = float(np.mean(scores))

        return results
    

    def temporal_tv(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        features: List[str],
        min_count_per_bin: int = 5,
    ) -> Dict:
        """
        시간 bin별 feature의 Total Variation 계산
        - Total Variation Distance = 0.5 * Σ|P(x) - Q(x)|
        
        for categorical features
        """
        real_binned = self.create_time_bins(real_df)
        syn_binned = self.create_time_bins(syn_df)

        results = {"per_feature": {}, "temporal_tv_overall": np.nan}
        scores = []

        for feat in features:
            if feat not in real_binned.columns or feat not in syn_binned.columns:
                continue

            common_bins = sorted(set(real_binned["time_bin"]).intersection(set(syn_binned["time_bin"])))
            if not common_bins:
                continue

            tv_stats = []
            for tb in common_bins:
                r = real_binned.loc[real_binned["time_bin"] == tb, feat].dropna()
                s = syn_binned.loc[syn_binned["time_bin"] == tb, feat].dropna()

                if len(r) < min_count_per_bin or len(s) < min_count_per_bin:
                    continue

                # Get frequency distributions
                f_real, f_syn = utils.get_frequencies(r.values, s.values)
                
                # Calculate Total Variation Distance
                total_variation = 0.5 * np.sum(np.abs(f_real - f_syn))
                tv_stats.append(total_variation)

            if tv_stats:
                feat_score = float(np.mean(tv_stats))
                results["per_feature"][feat] = feat_score
                scores.append(feat_score)

        if scores:
            results["temporal_tv_overall"] = float(np.mean(scores))

        return results

    def temporal_mmd(
        self,
        real_df: pd.DataFrame,
        syn_df: pd.DataFrame,
        features: List[str],
        kernel: str = "rbf",           # "rbf" | "linear" | "polynomial"
        gamma: float = 1.0,            # rbf/poly
        degree: int = 2,               # poly
        coef0: float = 0.0,            # poly
        min_count_per_bin: int = 10,
    ) -> Dict:
        """
        시간 bin별 joint MMD를 계산한 뒤 평균.
        - 각 time_bin에서 (n_bin_samples x d_features)로 구성된 행렬을 만들고
        - MMD = mean(Kxx) + mean(Kyy) - 2*mean(Kxy)
        
        for numerical features
        """
        real_binned = self.create_time_bins(real_df)
        syn_binned = self.create_time_bins(syn_df)

        use_feats = [f for f in features if f in real_binned.columns and f in syn_binned.columns]
        if not use_feats:
            return {"per_feature": {}, "overall": np.nan}
        
        for f in use_feats:
            real_binned[f] = pd.to_numeric(real_binned[f], errors="coerce")
            syn_binned[f] = pd.to_numeric(syn_binned[f], errors="coerce")

        common_bins = sorted(set(real_binned["time_bin"]).intersection(set(syn_binned["time_bin"])))
        if not common_bins:
            return {"per_feature": {}, "overall": np.nan}

        def _kernel(X, Y):
            if kernel == "linear":
                # linear_kernel은 X @ Y.T
                return linear_kernel(X, Y)
            elif kernel == "rbf":
                return rbf_kernel(X, Y, gamma=gamma)
            elif kernel == "polynomial":
                return polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)
            else:
                raise ValueError(f"Unsupported kernel {kernel}")

        mmds = []
        per_bin = {}

        for tb in common_bins:
            X = real_binned.loc[real_binned["time_bin"] == tb, use_feats].dropna().values
            Y = syn_binned.loc[syn_binned["time_bin"] == tb, use_feats].dropna().values

            if len(X) < min_count_per_bin or len(Y) < min_count_per_bin:
                continue

            Kxx = _kernel(X, X)
            Kyy = _kernel(Y, Y)
            Kxy = _kernel(X, Y)

            score = float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())
            per_bin[str(tb)] = score
            mmds.append(score)

        out = {
            "kernel": kernel,
            "params": {"gamma": gamma, "degree": degree, "coef0": coef0},
            "per_time_bin": per_bin,
            "temporal_mmd_overall": float(np.mean(mmds)) if mmds else np.nan,
        }
        return out

    def comprehensive_evaluation(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        features: List[str],
        num_cols,
        cat_cols,
        parent_key: Optional[str] = None,
    ) -> Dict:
        
        print("=" * 80)
        print("Temporal Benchmark Evaluation")
        print("=" * 80)
        print()
        
        # # ✅ DataFrame에서 직접 타입 추출
        # num_cols = self.get_numeric_columns(real_df, features)
        # cat_cols = self.get_categorical_columns(real_df, features)

        num_cols = num_cols
        cat_cols = cat_cols
        
        print(f"{'Numerical Columns':<22}({len(num_cols):>2}): {num_cols}")
        print(f"{'Categorical Columns':<22}({len(cat_cols):>2}): {cat_cols}")
        print()
        
        # 스케일링 (수치형만)
        real_df_scaled, synth_df_scaled = utils.scale_features(
            real_df.copy(), 
            synth_df.copy(), 
            num_cols  # 👈 수치형만 스케일링
        )

        results = {}
        
        # === 기본 메트릭 (전체 features) ===
        
        # # Bin Length Discrepancy
        # metrics_fld = self.bin_length_discrepancy(
        #     real_df, synth_df, features
        # )
        # results['bin_length_discrepancy'] = metrics_fld
        # print(f"Bin Length Discrepancy: {metrics_fld.get('bin_length_discrepancy_overall', np.nan):.4f}")


        if parent_key and parent_key in real_df.columns:
            print()
            
            # 전이 행렬 분석
            tm_results = transition_matrix_analysis(
                real_df, synth_df, parent_key, features, n_bins=5, time_column=self.time_column
            )
            results['transition_matrix'] = tm_results
            print(f"Transition Matrix L1: {tm_results.get('tm_l1_overall', np.nan):.4f}")
            print(f"Transition Matrix JSD: {tm_results.get('tm_jsd_overall', np.nan):.4f}")
            
            # lag-1 차분 분석 (수치형만)
            if num_cols:
                k = 1
                lag_results = lag_k_diff_analysis(
                    real_df_scaled, synth_df_scaled, parent_key, num_cols, k=k, time_column=self.time_column
                )
                results['lag_diff'] = lag_results
                print(f"Lag-{k} Diff KS: {lag_results.get('lag_diff_ks_overall', np.nan):.4f}")
                print(f"Lag-{k} Diff Wasserstein: {lag_results.get('lag_diff_wasserstein_overall', np.nan):.4f}")
                
                # acf_results = temporal_acf_comparison_2(
                #     real_df, synth_df, parent_key, num_cols, max_lag=14, time_column=self.time_column
                # )
                # results['acf'] = acf_results
                # print(f"ACF MAE: {acf_results.get('acf_mae_overall', np.nan):.4f}")
                # print(f"ACF Max Diff: {acf_results.get('acf_max_diff_overall', np.nan):.4f}")

        temporal_validity = evaluate_temporal_validity(
            real_df, synth_df, time_column=self.time_column
        )
        results['Temporal Validity'] = temporal_validity['support_jaccard']
        print(f"Temporal Validity: {temporal_validity['support_jaccard']:.4f} ")

        # # Temporal Mean/Var MAE (수치형만 의미있음)
        # if num_cols:
        #     metrics_mean = self.temporal_mean_mae(
        #         real_df_scaled, synth_df_scaled, num_cols
        #     )
        #     metrics_var = self.temporal_var_mae(
        #         real_df_scaled, synth_df_scaled, num_cols
        #     )
        #     results['temporal_mean_mae'] = metrics_mean
        #     results['temporal_var_mae'] = metrics_var
            
        #     print(f"Temporal MAE: {metrics_mean.get('temporal_mean_mae_overall', np.nan):.4f} "
        #           f"± {metrics_var.get('temporal_var_mae_overall', np.nan):.4f}")
        
        
        # === 범주형 메트릭 ===        
        if cat_cols:
            ## type casting
            for cols in cat_cols:
                synth_df[cols] = synth_df[cols].astype(real_df[cols].dtype)
        
            # JSD
            metrics_jsd = self.temporal_jsd(
                real_df, synth_df, cat_cols,
                base=2.0, min_count_per_bin=5
            )
            results['temporal_jsd'] = metrics_jsd
            print(f"Temporal JSD: {metrics_jsd.get('temporal_jsd_overall', np.nan):.4f}")
            
            # Total Variation
            metrics_tv = self.temporal_tv(
                real_df, synth_df, cat_cols, min_count_per_bin=5
            )
            results['temporal_tv'] = metrics_tv
            print(f"Temporal TV: {metrics_tv.get('temporal_tv_overall', np.nan):.4f}")
            
            
        # === 수치형 메트릭 ===        
        if num_cols:
            # Wasserstein Distance
            metrics_wass = self.temporal_wasserstein(
                real_df_scaled, synth_df_scaled, num_cols,
                min_count_per_bin=5
            )
            results['temporal_wasserstein'] = metrics_wass
            print(f"Temporal Wasserstein: {metrics_wass.get('temporal_wasserstein_overall', np.nan):.4f}")
            
            # KS
            metrics_ks = self.temporal_ks(
                real_df_scaled, synth_df_scaled, num_cols,
                min_count_per_bin=5
            )
            results['temporal_ks'] = metrics_ks
            print(f"Temporal KS: {metrics_ks.get('temporal_ks_overall', np.nan):.4f}")
            
            # MMD
            metrics_mmd = self.temporal_mmd(
                real_df_scaled, synth_df_scaled, num_cols,
                kernel="rbf", gamma=1.0, min_count_per_bin=10
            )
            results['temporal_mmd'] = metrics_mmd
            print(f"Temporal MMD ({metrics_mmd['kernel']}): {metrics_mmd.get('temporal_mmd_overall', np.nan):.4f}")
        
        print()
        print("=" * 80)
        
        return results