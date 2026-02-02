import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TemporalBenchmark:
    """
    시계열 데이터를 위한 벤치마크 프레임워크
    - Automatic change point detection
    - Time binning 기반 cross-sectional analysis
    - Concept drift detection
    """
    
    def __init__(
        self,
        time_column: str = 'Date',
        bin_strategy: str = 'monthly',
        custom_bins: Optional[List] = None
    ):
        self.time_column = time_column
        self.bin_strategy = bin_strategy
        self.custom_bins = custom_bins
        
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
        elif self.bin_strategy == 'custom' and self.custom_bins:
            df['time_bin'] = pd.cut(df[self.time_column], bins=self.custom_bins)
        else:
            raise ValueError(f"Unknown bin strategy: {self.bin_strategy}")
        
        return df
    
    def detect_change_points_window(
        self,
        df: pd.DataFrame,
        features: List[str],
        window_size: int = 3,
        threshold_percentile: float = 90
    ) -> Dict:
        """
        Rolling window 기반 change point 자동 탐지
        
        Args:
            df: 데이터프레임
            features: 분석할 feature 리스트
            window_size: 비교할 윈도우 크기 (bins 단위)
            threshold_percentile: 상위 몇 %를 change point로 볼 것인가
        
        Returns:
            탐지된 change points와 각 시점의 shift magnitude
        """
        df_binned = self.create_time_bins(df)
        time_bins = sorted(df_binned['time_bin'].unique())
        
        bin_counts = df_binned.groupby('time_bin').size().sort_index()
        # print(bin_counts)

        # print("time bins:\n", time_bins)
        
        if len(time_bins) < window_size * 2:
            return {'error': 'Not enough time bins for change point detection'}
        
        # 각 feature별로 shift 크기 계산
        shift_magnitudes = {feat: [] for feat in features}
        valid_transitions = []
        
        for i in range(len(time_bins) - window_size):
            window1_bins = time_bins[i:i+window_size]
            window2_bins = time_bins[i+window_size:i+window_size*2]
            
            # print(f"Comparing windows: {window1_bins} -> {window2_bins}")
            
            if len(window2_bins) < window_size:
                continue
            
            transition_point = time_bins[i+window_size]
            feature_shifts = []
            
            for feat in features:
                if feat not in df_binned.columns:
                    continue
                
                # Window 1 data
                w1_data = df_binned[df_binned['time_bin'].isin(window1_bins)][feat].dropna()
                
                # Window 2 data
                w2_data = df_binned[df_binned['time_bin'].isin(window2_bins)][feat].dropna()
                # if feat == 'Customers':
                #     print("w1_data:\n", w1_data)
                
                if len(w1_data) > 10 and len(w2_data) > 10:
                    # Wasserstein distance로 distribution shift 측정
                    shift = wasserstein_distance(w1_data.values, w2_data.values)
                    shift_magnitudes[feat].append(shift)
                    feature_shifts.append(shift)
            
            if feature_shifts:
                valid_transitions.append({
                    'transition_bin': transition_point,
                    'mean_shift': np.mean(feature_shifts),
                    'max_shift': np.max(feature_shifts)
                })
        
        # Threshold 계산: 상위 percentile
        all_shifts = [t['mean_shift'] for t in valid_transitions]
        if not all_shifts:
            return {'change_points': [], 'all_transitions': []}
        
        threshold = np.percentile(all_shifts, threshold_percentile)
        
        # print("\n", all_shifts)
        # print("all_shifts length:", len(all_shifts))
        # print("Threshold for change points ("f"{threshold_percentile}th percentile): {threshold:.4f} \n")
        
        # Change points 필터링
        change_points = [
            t for t in valid_transitions 
            if t['mean_shift'] >= threshold
        ]
        
        # 시간순 정렬
        change_points.sort(key=lambda x: x['transition_bin'])
        
        return {
            'change_points': change_points,
            'all_transitions': valid_transitions,
            'threshold': threshold,
            'shift_magnitudes_by_feature': shift_magnitudes
        }
    
    def detect_change_points_zscore(
        self,
        df: pd.DataFrame,
        features: List[str],
        z_threshold: float = 2.0
    ) -> Dict:
        """
        Z-score 기반 change point 탐지
        각 시간 bin의 평균이 전체 평균에서 얼마나 벗어나는지
        """
        df_binned = self.create_time_bins(df)
        time_bins = sorted(df_binned['time_bin'].unique())
        
        change_points_by_feature = {}
        
        for feat in features:
            if feat not in df_binned.columns:
                continue
            
            # 각 bin의 평균 계산
            bin_means = df_binned.groupby('time_bin')[feat].mean()
            
            # Z-score 계산
            mean = bin_means.mean()
            std = bin_means.std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((bin_means - mean) / std)
            
            # Threshold 넘는 bins
            anomalous_bins = z_scores[z_scores > z_threshold].index.tolist()
            
            change_points_by_feature[feat] = {
                'anomalous_bins': anomalous_bins,
                'z_scores': z_scores.to_dict()
            }
        
        # 여러 feature에서 공통으로 나타나는 시점
        all_anomalous = []
        for feat_data in change_points_by_feature.values():
            all_anomalous.extend(feat_data['anomalous_bins'])
        
        from collections import Counter
        bin_counts = Counter(all_anomalous)
        
        # 2개 이상의 feature에서 anomaly로 감지된 시점
        common_change_points = [
            bin_id for bin_id, count in bin_counts.items() 
            if count >= min(2, len(features))
        ]
        
        return {
            'common_change_points': sorted(common_change_points),
            'by_feature': change_points_by_feature
        }
    
    def compare_change_points(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        features: List[str],
        method: str = 'window',  # 'window' or 'zscore'
        **kwargs
    ) -> Dict:
        """
        Real과 Synthetic 데이터에서 각각 change point를 찾고 비교
        
        Returns:
            - 각각의 change points
            - Overlap (얼마나 겹치는지)
            - Temporal alignment (시간적으로 얼마나 가까운지)
        """
        print(f"\n[Change Point Detection] Using method: {method}")
        
        if method == 'window':
            real_cp = self.detect_change_points_window(real_df, features, **kwargs)
            synth_cp = self.detect_change_points_window(synth_df, features, **kwargs)
            
            if 'error' in real_cp or 'error' in synth_cp:
                return {'error': 'Not enough data for change point detection'}
            
            real_points = [cp['transition_bin'] for cp in real_cp['change_points']]
            synth_points = [cp['transition_bin'] for cp in synth_cp['change_points']]
            
        elif method == 'zscore':
            real_cp = self.detect_change_points_zscore(real_df, features, **kwargs)
            synth_cp = self.detect_change_points_zscore(synth_df, features, **kwargs)
            
            real_points = real_cp['common_change_points']
            synth_points = synth_cp['common_change_points']
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"   Real data: {len(real_points)} change points detected")
        print(f"   Synthetic data: {len(synth_points)} change points detected")
        
        # Exact overlap
        real_set = set(real_points)
        synth_set = set(synth_points)
        overlap = real_set.intersection(synth_set)
        
        # Jaccard similarity
        union = real_set.union(synth_set)
        jaccard = len(overlap) / len(union) if len(union) > 0 else 0
        
        # Temporal alignment: 가까운 시점끼리 매칭
        if real_points and synth_points:
            # Period string을 datetime으로 변환
            def parse_period_string(period_str):
                """Period string을 datetime으로 변환"""
                # Weekly format: "2014-09-22/2014-09-28" → 시작 날짜만 추출
                if '/' in period_str:
                    return pd.to_datetime(period_str.split('/')[0])
                # Monthly/Quarterly format: "2014-09" or "2014Q3" → Period로 변환 후 start_time
                else:
                    try:
                        period = pd.Period(period_str)
                        return period.to_timestamp()
                    except:
                        # Fallback: 그냥 파싱 시도
                        return pd.to_datetime(period_str)
            
            try:
                real_dates = [parse_period_string(p) for p in real_points]
                synth_dates = [parse_period_string(p) for p in synth_points]
                
                temporal_distances = []
                for real_date in real_dates:
                    min_distance = min([abs((real_date - synth_date).days) for synth_date in synth_dates])
                    temporal_distances.append(min_distance)
                
                mean_temporal_distance = np.mean(temporal_distances)
            except Exception as e:
                print(f"   Warning: Could not compute temporal distance: {e}")
                mean_temporal_distance = None
        else:
            mean_temporal_distance = None
        
        result = {
            'real_change_points': real_points,
            'synth_change_points': synth_points,
            'exact_overlap': list(overlap),
            'jaccard_similarity': jaccard,
            'mean_temporal_distance_days': mean_temporal_distance,
            'real_cp_details': real_cp,
            'synth_cp_details': synth_cp
        }
        
        print(f"   Jaccard similarity: {jaccard:.4f}")
        print(f"   Exact overlap: {len(overlap)} bins")
        if mean_temporal_distance is not None:
            print(f"   Mean temporal distance: {mean_temporal_distance:.1f} days")
        
        return result
    
    def auto_event_shift_analysis(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        features: List[str],
        change_points: List[str],
        before_after_bins: int = 2
    ) -> Dict:
        """
        자동으로 탐지된 change points에서 shift 분석
        
        Args:
            change_points: 분석할 change point 리스트 (time bin strings)
            before_after_bins: 전후로 몇 개 bin을 볼 것인가
        """
        real_binned = self.create_time_bins(real_df)
        synth_binned = self.create_time_bins(synth_df)
        
        time_bins = sorted(real_binned['time_bin'].unique())
        
        results = []
        
        for cp in change_points:
            if cp not in time_bins:
                continue
            
            cp_idx = time_bins.index(cp)
            
            # Before bins
            before_start = max(0, cp_idx - before_after_bins)
            before_bins = time_bins[before_start:cp_idx]
            
            # print("before bins\n", before_bins)
            
            # After bins
            after_end = min(len(time_bins), cp_idx + before_after_bins)
            after_bins = time_bins[cp_idx:after_end]
            
            # print("after bins\n", after_bins)
            
            if len(before_bins) == 0 or len(after_bins) == 0:
                continue
            
            # Real data
            real_before = real_binned[real_binned['time_bin'].isin(before_bins)]
            real_after = real_binned[real_binned['time_bin'].isin(after_bins)]
            
            # Synthetic data
            synth_before = synth_binned[synth_binned['time_bin'].isin(before_bins)]
            synth_after = synth_binned[synth_binned['time_bin'].isin(after_bins)]
            
            shift_fidelity_scores = {}
            
            for feat in features:
                if feat not in real_before.columns:
                    continue
                
                # Real shift
                real_shift = wasserstein_distance(
                    real_before[feat].dropna().values,
                    real_after[feat].dropna().values
                )
                
                # Synthetic shift
                synth_shift = wasserstein_distance(
                    synth_before[feat].dropna().values,
                    synth_after[feat].dropna().values
                )
                
                # Fidelity
                if max(real_shift, synth_shift) > 0:
                    fidelity = 1 - abs(real_shift - synth_shift) / max(real_shift, synth_shift)
                else:
                    fidelity = 1.0
                
                shift_fidelity_scores[feat] = {
                    'real_shift': float(real_shift),
                    'synth_shift': float(synth_shift),
                    'fidelity': float(fidelity)
                }
            
            overall_fidelity = np.mean([
                scores['fidelity'] for scores in shift_fidelity_scores.values()
            ]) if shift_fidelity_scores else 0.0
            
            results.append({
                'change_point_bin': cp,
                'shift_fidelity_scores': shift_fidelity_scores,
                'overall_shift_fidelity': float(overall_fidelity)
            })
        
        return {
            'per_change_point': results,
            'mean_fidelity': np.mean([r['overall_shift_fidelity'] for r in results]) if results else 0.0
        }
    
    def temporal_cross_sectional_fidelity(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        features: List[str],
        method: str = 'wasserstein',
        normalize: bool = True  # ← 추가
    ) -> Dict:
        """시간 구간별로 분포 유사도 측정"""
        real_binned = self.create_time_bins(real_df)
        synth_binned = self.create_time_bins(synth_df)
        
        time_bins = sorted(real_binned['time_bin'].unique())
        
        # Feature별 scale 미리 계산
        feature_scales = {}
        if normalize:
            for feat in features:
                scale = real_df[feat].std()
                feature_scales[feat] = scale if scale > 0 else 1.0
        
        results = {
            'per_bin_scores': {},
            'per_feature_scores': {feat: [] for feat in features},
            'per_feature_scores_raw': {feat: [] for feat in features},  # Raw 값도 저장
            'overall_score': 0.0
        }
        
        all_scores = []
        
        for bin_id in time_bins:
            real_bin = real_binned[real_binned['time_bin'] == bin_id]
            synth_bin = synth_binned[synth_binned['time_bin'] == bin_id]
            
            if len(real_bin) == 0 or len(synth_bin) == 0:
                continue
            
            bin_scores = []
            
            for feat in features:
                if feat not in real_bin.columns or feat not in synth_bin.columns:
                    continue
                
                raw_score = self._compute_distribution_distance(
                    real_bin[feat].dropna().values,
                    synth_bin[feat].dropna().values,
                    method=method
                )
                
                # Normalize
                if normalize and feat in feature_scales:
                    score = raw_score / feature_scales[feat]
                else:
                    score = raw_score
                
                bin_scores.append(score)
                results['per_feature_scores'][feat].append(score)
                results['per_feature_scores_raw'][feat].append(raw_score)
            
            if bin_scores:
                results['per_bin_scores'][bin_id] = np.mean(bin_scores)
                all_scores.extend(bin_scores)
        
        results['overall_score'] = np.mean(all_scores) if all_scores else 0.0
        results['score_std'] = np.std(all_scores) if all_scores else 0.0
        
        # Feature별 평균
        for feat in features:
            if results['per_feature_scores'][feat]:
                results['per_feature_scores'][feat] = np.mean(results['per_feature_scores'][feat])
                results['per_feature_scores_raw'][feat] = np.mean(results['per_feature_scores_raw'][feat])
            else:
                results['per_feature_scores'][feat] = 0.0
                results['per_feature_scores_raw'][feat] = 0.0
        
        return results

    def concept_drift_correlation(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        features: List[str],
        window_size: int = 5
    ) -> Dict:
        """시간에 따른 concept drift 패턴이 얼마나 유사한지"""
        real_binned = self.create_time_bins(real_df)
        synth_binned = self.create_time_bins(synth_df)
        
        time_bins = sorted(real_binned['time_bin'].unique())
        
        if len(time_bins) < window_size + 1:
            return {'error': 'Not enough time bins for drift analysis'}
        
        results = {
            'drift_correlations': {},
            'overall_correlation': 0.0
        }
        
        for feat in features:
            if feat not in real_binned.columns:
                continue
            
            real_drifts = []
            synth_drifts = []
            
            for i in range(len(time_bins) - window_size):
                window1_bins = time_bins[i:i+window_size]
                window2_bins = time_bins[i+1:i+window_size+1]
                
                # print(f"Comparing windows: {window1_bins} -> {window2_bins}")
                
                real_w1 = real_binned[real_binned['time_bin'].isin(window1_bins)][feat].dropna()
                synth_w1 = synth_binned[synth_binned['time_bin'].isin(window1_bins)][feat].dropna()
                
                real_w2 = real_binned[real_binned['time_bin'].isin(window2_bins)][feat].dropna()
                synth_w2 = synth_binned[synth_binned['time_bin'].isin(window2_bins)][feat].dropna()
                
                # ⭐ 둘 다 유효한 경우에만 drift 계산 및 추가
                if (len(real_w1) > 0 and len(real_w2) > 0 and 
                    len(synth_w1) > 0 and len(synth_w2) > 0):
                    
                    real_drift = wasserstein_distance(real_w1.values, real_w2.values)
                    synth_drift = wasserstein_distance(synth_w1.values, synth_w2.values)
                    
                    real_drifts.append(real_drift)
                    synth_drifts.append(synth_drift)
            
            # ⭐ 길이 체크 추가
            if len(real_drifts) > 1 and len(synth_drifts) > 1 and len(real_drifts) == len(synth_drifts):
                correlation = np.corrcoef(real_drifts, synth_drifts)[0, 1]
                results['drift_correlations'][feat] = float(correlation)
            elif len(real_drifts) != len(synth_drifts):
                print(f"   Warning: Drift list length mismatch for {feat} "
                    f"(real: {len(real_drifts)}, synth: {len(synth_drifts)})")
        
        if results['drift_correlations']:
            results['overall_correlation'] = np.mean(list(results['drift_correlations'].values()))
        
        return results
    
    def temporal_autocorrelation_fidelity(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        features: List[str],
        max_lag: int = 10
    ) -> Dict:
        """시간별 자기상관 구조가 얼마나 보존되는지"""
        real_binned = self.create_time_bins(real_df)
        synth_binned = self.create_time_bins(synth_df)
                
        results = {
            'acf_mae': {},
            'overall_acf_mae': 0.0
        }
        
        for feat in features:
            if feat not in real_binned.columns:
                continue
            
            real_ts = real_binned.groupby('time_bin')[feat].mean().sort_index()
            synth_ts = synth_binned.groupby('time_bin')[feat].mean().sort_index()
                        
            if len(real_ts) < max_lag + 1 or len(synth_ts) < max_lag + 1:
                continue
            
            real_acf = [real_ts.autocorr(lag=i) for i in range(1, min(max_lag, len(real_ts)))]
            synth_acf = [synth_ts.autocorr(lag=i) for i in range(1, min(max_lag, len(synth_ts)))]
            
            # print("real acf:\n", real_acf)
            
            real_acf = [x for x in real_acf if not np.isnan(x)]
            synth_acf = [x for x in synth_acf if not np.isnan(x)]
            
            if real_acf and synth_acf:
                mae = np.mean(np.abs(np.array(real_acf) - np.array(synth_acf)))
                results['acf_mae'][feat] = float(mae)
        
        if results['acf_mae']:
            results['overall_acf_mae'] = np.mean(list(results['acf_mae'].values()))
        
        return results
    
    def _compute_distribution_distance(
        self,
        real_values: np.ndarray,
        synth_values: np.ndarray,
        method: str = 'wasserstein'
    ) -> float:
        """분포 간 거리 계산"""
        if len(real_values) == 0 or len(synth_values) == 0:
            return 0.0
        
        if method == 'wasserstein':
            return wasserstein_distance(real_values, synth_values)
        
        elif method == 'ks':
            statistic, _ = ks_2samp(real_values, synth_values)
            return 1 - statistic
        
        elif method == 'kl' or method == 'js':
            all_values = np.concatenate([real_values, synth_values])
            bins = np.histogram_bin_edges(all_values, bins='auto')
            
            real_hist, _ = np.histogram(real_values, bins=bins, density=True)
            synth_hist, _ = np.histogram(synth_values, bins=bins, density=True)
            
            real_hist = real_hist / (real_hist.sum() + 1e-10)
            synth_hist = synth_hist / (synth_hist.sum() + 1e-10)
            
            real_hist += 1e-10
            synth_hist += 1e-10
            
            if method == 'kl':
                kl = np.sum(real_hist * np.log(real_hist / synth_hist))
                return np.exp(-kl)
            else:
                js = jensenshannon(real_hist, synth_hist)
                return 1 - js
        
        return 0.0
    
    def comprehensive_evaluation(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        features: List[str],
        auto_detect_events: bool = True,
        detection_method: str = 'window',  # 'window' or 'zscore'
        **detection_kwargs
    ) -> Dict:
        """종합 평가 with automatic change point detection"""
        print("=" * 80)
        print("Temporal Benchmark Evaluation (with Auto Change Point Detection)")
        print("=" * 80)
        
        results = {}
        
        # 1. Temporal Cross-sectional Fidelity
        print("\n[1/5] Computing Temporal Cross-sectional Fidelity...")
        cross_sectional = self.temporal_cross_sectional_fidelity(
            real_df, synth_df, features, method='wasserstein'
        )
        results['cross_sectional_fidelity'] = cross_sectional
        print(f"   Overall Score: {cross_sectional['overall_score']:.4f} ± {cross_sectional['score_std']:.4f}")
        
        # 2. Automatic Change Point Detection & Comparison
        if auto_detect_events:
            print("\n[2/5] Detecting Change Points Automatically...")
            cp_comparison = self.compare_change_points(
                real_df, synth_df, features,
                method=detection_method,
                **detection_kwargs
            )
            results['change_point_comparison'] = cp_comparison
            
            # 3. Event-triggered Shift Analysis at detected change points
            if cp_comparison.get('exact_overlap'):
                print("\n[3/5] Computing Shift Analysis at Detected Change Points...")
                event_shift = self.auto_event_shift_analysis(
                    real_df, synth_df, features,
                    change_points=cp_comparison['exact_overlap']
                )
                results['auto_event_shift_analysis'] = event_shift
                print(f"   Mean Shift Fidelity: {event_shift['mean_fidelity']:.4f}")
            else:
                print("\n[3/5] No overlapping change points detected, skipping shift analysis")
                results['auto_event_shift_analysis'] = None
        
        # 4. Concept Drift Correlation
        print("\n[4/5] Computing Concept Drift Correlation...")
        drift_corr = self.concept_drift_correlation(
            real_df, synth_df, features, window_size=3
        )
        results['drift_correlation'] = drift_corr
        if 'overall_correlation' in drift_corr:
            print(f"   Overall Correlation: {drift_corr['overall_correlation']:.4f}")
        
        # 5. Temporal Autocorrelation Fidelity
        print("\n[5/5] Computing Temporal Autocorrelation Fidelity...")
        acf_fidelity = self.temporal_autocorrelation_fidelity(
            real_df, synth_df, features, max_lag=5
        )
        results['autocorrelation_fidelity'] = acf_fidelity
        if 'overall_acf_mae' in acf_fidelity:
            print(f"   Overall ACF MAE: {acf_fidelity['overall_acf_mae']:.4f}")
        
        print("\n" + "=" * 80)
        print("Evaluation Complete!")
        print("=" * 80)
        
        return results


# ============================================================================
# Rossmann-specific 사용 예시
# ============================================================================

def run_rossmann_temporal_benchmark(
    real_path: str = 'data/rossmann_subsampled_real.csv',
    synth_path: str = 'data/rossmann_subsampled_synthetic.csv'
):
    """
    Rossmann 데이터로 temporal benchmark 실행 (자동 change point detection)
    """
    print("Loading Rossmann data...")
    real_df = pd.read_csv(real_path)
    synth_df = pd.read_csv(synth_path)
    
    # Date 컬럼을 datetime으로 변환
    real_df['Date'] = pd.to_datetime(real_df['Date'])
    synth_df['Date'] = pd.to_datetime(synth_df['Date'])
    
    real_df = real_df.sort_values('Date').reset_index(drop=True)
    synth_df = synth_df.sort_values('Date').reset_index(drop=True)
    
    print(f"Real data shape: {real_df.shape}")
    print(f"Synthetic data shape: {synth_df.shape}")
    print(f"Date range (real): {real_df['Date'].min()} to {real_df['Date'].max()}")
    print(f"Date range (synth): {synth_df['Date'].min()} to {synth_df['Date'].max()}")
    
    # Rossmann 실제 컬럼 기준 numerical features
    available_features = []
    candidate_features = ['Customers', 'Open', 'Promo', 'DayOfWeek']

    for feat in candidate_features:
        if feat in real_df.columns and feat in synth_df.columns:
            # 이미 numeric dtype이면 바로 추가
            if pd.api.types.is_numeric_dtype(real_df[feat]) and pd.api.types.is_numeric_dtype(synth_df[feat]):
                available_features.append(feat)
            else:
                # Categorical이지만 값이 numeric이면 변환해서 사용
                try:
                    real_numeric = pd.to_numeric(real_df[feat], errors='coerce')
                    synth_numeric = pd.to_numeric(synth_df[feat], errors='coerce')
                    
                    # 변환이 성공적이면 (NaN이 너무 많지 않으면)
                    if real_numeric.notna().mean() > 0.9 and synth_numeric.notna().mean() > 0.9:
                        # 원본 데이터프레임도 변환
                        print(f"Converting feature '{feat}' to numeric for analysis")
                        real_df[feat] = real_numeric
                        synth_df[feat] = synth_numeric
                        available_features.append(feat)
                except:
                    pass
    
    print(f"Available numerical features: {available_features}")
    
    if not available_features:
        print("Error: No common numerical features found!")
        return None
    
    # 월별 binning
    benchmark = TemporalBenchmark(
        time_column='Date',
        bin_strategy='daily'
    )
    
    # 종합 평가 with automatic event detection
    results = benchmark.comprehensive_evaluation(
        real_df=real_df,
        synth_df=synth_df,
        features=available_features,
        auto_detect_events=True,
        detection_method='window',  # 'window' or 'zscore'
        window_size=2,  # window-based method용
        threshold_percentile=85  # 상위 15%를 change point로
    )
    
    # 결과 출력
    print("\n\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    
    print("\n1. Cross-sectional Fidelity (per feature):")
    for feat, score in results['cross_sectional_fidelity']['per_feature_scores'].items():
        print(f"   {feat}: {score:.4f}")
    
    if 'change_point_comparison' in results and 'error' not in results['change_point_comparison']:
        print("\n2. Change Point Detection:")
        cp_comp = results['change_point_comparison']
        print(f"   Real change points: {len(cp_comp['real_change_points'])}")
        print(f"   Detected points: {cp_comp['real_change_points']}")
        print(f"   Synthetic change points: {len(cp_comp['synth_change_points'])}")
        print(f"   Detected points: {cp_comp['synth_change_points']}")
        print(f"   Jaccard similarity: {cp_comp['jaccard_similarity']:.4f}")
        if cp_comp['mean_temporal_distance_days'] is not None:
            print(f"   Mean temporal distance: {cp_comp['mean_temporal_distance_days']:.1f} days")
    
    if 'auto_event_shift_analysis' in results and results['auto_event_shift_analysis']:
        print("\n3. Auto Event Shift Fidelity:")
        print(f"   Mean fidelity: {results['auto_event_shift_analysis']['mean_fidelity']:.4f}")
        print(f"   Number of events analyzed: {len(results['auto_event_shift_analysis']['per_change_point'])}")
        
        # 각 change point별 상세 결과
        for cp_result in results['auto_event_shift_analysis']['per_change_point']:
            print(f"\n   Change Point: {cp_result['change_point_bin']}")
            print(f"   Overall Fidelity: {cp_result['overall_shift_fidelity']:.4f}")
            for feat, scores in cp_result['shift_fidelity_scores'].items():
                print(f"      {feat}: real_shift={scores['real_shift']:.2f}, "
                      f"synth_shift={scores['synth_shift']:.2f}, fidelity={scores['fidelity']:.4f}")
    
    if 'drift_correlation' in results and 'drift_correlations' in results['drift_correlation']:
        print("\n4. Drift Correlation (per feature):")
        for feat, corr in results['drift_correlation']['drift_correlations'].items():
            print(f"   {feat}: {corr:.4f}")
    
    if 'autocorrelation_fidelity' in results and 'acf_mae' in results['autocorrelation_fidelity']:
        print("\n5. ACF MAE (per feature):")
        for feat, mae in results['autocorrelation_fidelity']['acf_mae'].items():
            print(f"   {feat}: {mae:.4f}")
    
    return results


if __name__ == "__main__":
    
    #                 0           1             2           3        4        5 
    syn_models = ['CLAVADDPM', 'RCTGAN', 'REALTABFORMER', 'RGCLD', 'SDV', 'RelDiff']
    syn_model_name = syn_models[5]
    
    # 실행
    results = run_rossmann_temporal_benchmark(
        real_path = "/home/yjung/syntherela/experiments/data/original/rossmann_subsampled/historical.csv",
        synth_path  = f"/home/yjung/syntherela/experiments/data/synthetic/rossmann_subsampled/{syn_model_name}/1/sample1/historical.csv",
    )
    
    if results:
        # 결과 저장
        import json
        with open('temporal_benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("\n✅ Results saved to temporal_benchmark_results.json")
