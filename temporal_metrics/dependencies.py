import pandas as pd
import numpy as np 
from typing import Dict, List, Optional

from statsmodels.tsa.stattools import acf
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp

def transition_matrix_analysis(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    parent_key: str,
    features: List[str],
    n_bins: int = 5,
    time_column: Optional[str] = None,
) -> Dict:
    """
    개체별 전이 행렬(Transition Matrix) 기반 시계열 분석
    - parent_key로 그룹화하여 각 개체의 상태 전이를 추적
    - 수치형: n_bins로 구간화, 범주형: 그대로 사용
    """
    
    def _calculate_transition_matrix(df, parent_key, state_col, datetime_col=None, n_bins=5):
        """단일 컬럼에 대한 전이 행렬 계산"""
        df_tm = df.copy()
        
        if df_tm.empty or state_col not in df_tm.columns:
            return pd.DataFrame()
        
        valid_values = df_tm[state_col].dropna()
        if len(valid_values) == 0:
            return pd.DataFrame()
        
        # 상태 컬럼이 수치형이면 구간화
        if pd.api.types.is_numeric_dtype(df_tm[state_col]):
            if len(valid_values) < 2:
                return pd.DataFrame()
            try:
                df_tm['state'] = pd.cut(df_tm[state_col], bins=n_bins, labels=False, include_lowest=True)
                labels = range(n_bins)
            except ValueError:
                return pd.DataFrame()
        else:  # 범주형
            df_tm['state'] = df_tm[state_col].astype('category')
            labels = df_tm['state'].cat.categories
            if len(labels) == 0:
                return pd.DataFrame()
        
        transitions = []
        
        if datetime_col is None or datetime_col not in df_tm.columns:
            # datetime 없으면 parent_key로만 정렬
            df_tm = df_tm.sort_values([parent_key]).reset_index(drop=True)
        else:
            # datetime 있으면 parent_key + datetime 정렬
            df_tm = df_tm.sort_values([parent_key, datetime_col]).reset_index(drop=True)
        
        for _, group in df_tm.groupby(parent_key):
            if len(group) < 2:
                continue
            current_states = group['state'][:-1]
            next_states = group['state'][1:]
            transitions.extend(zip(current_states, next_states))
            
        if not transitions:
            return pd.DataFrame()
        
        try:
            counts = pd.crosstab(
                pd.Series([t[0] for t in transitions], name='current_state'),
                pd.Series([t[1] for t in transitions], name='next_state')
            ).reindex(index=labels, columns=labels, fill_value=0)
            
            # 확률로 정규화
            tm = counts.div(counts.sum(axis=1), axis=0).fillna(0)
            return tm
        except Exception:
            return pd.DataFrame()
    
    def _transition_diff_L1(real_tm, syn_tm):
        """전이 행렬 간 L1 거리"""
        real_tm, syn_tm = real_tm.fillna(0), syn_tm.fillna(0).reindex_like(real_tm)
        return np.mean(np.abs(real_tm.values - syn_tm.values))
    
    def _transition_diff_JSD(real_tm, syn_tm):
        """전이 행렬 간 평균 JSD"""
        real_tm, syn_tm = real_tm.fillna(0), syn_tm.fillna(0).reindex_like(real_tm)
        
        eps = 1e-10
        real_vals = real_tm.values + eps
        syn_vals = syn_tm.values + eps
        
        real_norm = real_vals / real_vals.sum(axis=1, keepdims=True)
        syn_norm = syn_vals / syn_vals.sum(axis=1, keepdims=True)
        
        jsd_values = []
        for i in range(real_norm.shape[0]):
            jsd = float(jensenshannon(real_norm[i], syn_norm[i]))
            jsd_values.append(jsd)
        
        return np.mean(jsd_values)
    
    # 메인 로직
    results = {
        "per_feature": {},
        "tm_l1_overall": np.nan,
        "tm_jsd_overall": np.nan
    }
    
    l1_scores = []
    jsd_scores = []
    
    for feat in features:
        if feat not in real_df.columns or feat not in syn_df.columns:
            continue
        
        # 전이 행렬 계산
        real_tm = _calculate_transition_matrix(
            real_df, parent_key, feat, time_column, n_bins
        )
        syn_tm = _calculate_transition_matrix(
            syn_df, parent_key, feat, time_column, n_bins
        )
        
        if real_tm.empty or syn_tm.empty:
            continue
        
        # 메트릭 계산
        l1 = _transition_diff_L1(real_tm, syn_tm)
        jsd = _transition_diff_JSD(real_tm, syn_tm)
        
        results["per_feature"][feat] = {
            "tm_l1": l1,
            "tm_jsd": jsd
        }
        
        l1_scores.append(l1)
        jsd_scores.append(jsd)
    
    if l1_scores:
        results["tm_l1_overall"] = float(np.mean(l1_scores))
    if jsd_scores:
        results["tm_jsd_overall"] = float(np.mean(jsd_scores))
    
    return results


def lag_k_diff_analysis(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    parent_key: str,
    features: List[str],
    k: int = 1,
    time_column: Optional[str] = None,
) -> Dict:
    """
    개체별 lag-k 차분 분포 분석
    - 각 개체(parent_key)별로 시계열 차분을 계산
    - KS 테스트와 JSD로 차분 분포의 유사도 평가
    """
    
    def _calculate_lag_k_diffs(df, parent_key, datetime_col, numeric_col, k=1):
        """lag-k 차분 계산"""
        all_diffs = []
        
        if df.empty or numeric_col not in df.columns or parent_key not in df.columns:
            return np.array([])
        
        if datetime_col is None or datetime_col not in df.columns:
            df_sorted = df.sort_values([parent_key]).reset_index(drop=True)
        else:
            df_sorted = df.sort_values([parent_key, datetime_col]).reset_index(drop=True)
        
        for _, group in df_sorted.groupby(parent_key):
            if len(group) < k + 1:
                continue
            
            values = group[numeric_col].dropna()
            if len(values) < k + 1:
                continue
            
            # lag-k 차분
            diffs = values.diff(periods=k).dropna()
            all_diffs.extend(diffs.tolist())
            
        return np.array(all_diffs)
    
    def _lag_diff_metrics(real_diffs, syn_diffs):
        """차분 분포에 대한 KS 테스트와 WD"""
        if len(real_diffs) == 0 or len(syn_diffs) == 0:
            return np.nan, np.nan, np.nan
        
        # KS 테스트
        ks_stat, p_value = ks_2samp(real_diffs, syn_diffs)

        # Wasserstein Distance (연속형 데이터에 적합)
        wd = float(wasserstein_distance(real_diffs, syn_diffs))
        
        return float(ks_stat), float(p_value), wd
    
    # 메인 로직
    results = {
        "per_feature": {},
        "lag_diff_ks_overall": np.nan,
        "lag_diff_wasserstein_overall": np.nan
    }
    
    ks_scores = []
    wd_scores = []
    
    for feat in features:
        if feat not in syn_df.columns:
            continue
        
        # lag-k 차분 계산
        real_diffs = _calculate_lag_k_diffs(
            real_df, parent_key, time_column, feat, k
        )
        syn_diffs = _calculate_lag_k_diffs(
            syn_df, parent_key, time_column, feat, k
        )
        
        if len(real_diffs) == 0 or len(syn_diffs) == 0:
            continue
        
        # 메트릭 계산
        ks_stat, p_value, wd = _lag_diff_metrics(real_diffs, syn_diffs)

        print(f"syn_diffs 길이: {len(syn_diffs)}")
        
        results["per_feature"][feat] = {
            "ks_stat": ks_stat,
            "p_value": p_value,
            "wasserstein_distance": wd
        }
        
        if not np.isnan(ks_stat):
            ks_scores.append(ks_stat)
        if not np.isnan(wd):
            wd_scores.append(wd)
    
    if ks_scores:
        results["lag_diff_ks_overall"] = float(np.mean(ks_scores))
    if wd_scores:
        results["lag_diff_wasserstein_overall"] = float(np.mean(wd_scores))
    
    return results

def temporal_acf_comparison(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    parent_key: str,
    features: List[str],
    max_lag: int = 10,
    min_length: int = 30,
    time_column: Optional[str] = None,
) -> Dict:
    """
    개체별 ACF(자기상관함수) 패턴 비교
    - 각 개체(parent_key)별로 시계열의 자기상관 계산
    - 실제 데이터와 합성 데이터의 ACF 패턴 차이를 MAE로 측정
    
    Args:
        max_lag: 계산할 최대 lag
        min_length: ACF 계산에 필요한 최소 시계열 길이
    """
    
    def _calculate_acf_per_entity(df, parent_key, datetime_col, numeric_col, max_lag):
        """각 개체별 ACF를 계산하고 평균"""
        all_acfs = []
        
        if df.empty or numeric_col not in df.columns or parent_key not in df.columns:
            return None
        
        # 정렬
        if datetime_col is None or datetime_col not in df.columns:
            df_sorted = df.sort_values([parent_key]).reset_index(drop=True)
        else:
            df_sorted = df.sort_values([parent_key, datetime_col]).reset_index(drop=True)
        
        for _, group in df_sorted.groupby(parent_key):
            values = group[numeric_col].dropna()

            # 최소 길이 체크 (ACF 계산을 위해)
            if len(values) < min_length:
                continue
            
            try:
                # ACF 계산 (lag 0 제외)
                acf_values = acf(values, nlags=max_lag, fft=True)[1:]  # lag 0는 항상 1이므로 제외
                all_acfs.append(acf_values)
            except Exception as e:
                # 상수 시계열 등의 경우 에러 발생 가능
                continue
        
        if not all_acfs:
            return None
        
        # 모든 개체의 ACF를 평균
        mean_acf = np.mean(all_acfs, axis=0)
        return mean_acf
    
    # 메인 로직
    results = {
        "per_feature": {},
        "acf_mae_overall": np.nan,
        "acf_max_diff_overall": np.nan,
    }
    
    mae_scores = []
    max_diff_scores = []
    
    for feat in features:
        if feat not in syn_df.columns:
            continue
        
        # 실제/합성 데이터의 평균 ACF 계산
        real_acf = _calculate_acf_per_entity(
            real_df, parent_key, time_column, feat, max_lag
        )
        syn_acf = _calculate_acf_per_entity(
            syn_df, parent_key, time_column, feat, max_lag
        )
        
        if real_acf is None or syn_acf is None:
            continue
        
        # ACF 패턴 차이 계산
        acf_diff = np.abs(real_acf - syn_acf)
        mae = float(np.mean(acf_diff))
        max_diff = float(np.max(acf_diff))
        
        results["per_feature"][feat] = {
            "acf_mae": mae,
            "acf_max_diff": max_diff,
            "real_acf": real_acf.tolist(),
            "syn_acf": syn_acf.tolist(),
        }
        
        mae_scores.append(mae)
        max_diff_scores.append(max_diff)
    
    if mae_scores:
        results["acf_mae_overall"] = float(np.mean(mae_scores))
    if max_diff_scores:
        results["acf_max_diff_overall"] = float(np.mean(max_diff_scores))
    
    return results


def temporal_acf_comparison_2(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    parent_key: str,
    features: List[str],
    max_lag: int = 14,
    min_length: int = 5,
    time_column: Optional[str] = None,
) -> Dict:
    """
    개체별 ACF(자기상관함수) 패턴 비교
    """
    
    def _calculate_acf_per_entity(df, parent_key, datetime_col, numeric_col, max_lag, min_length):
        """각 개체별 ACF를 계산하고 평균"""
        all_acfs = []
        
        if df.empty or numeric_col not in df.columns or parent_key not in df.columns:
            return None
        
        # 정렬
        if datetime_col is None or datetime_col not in df.columns:
            df_sorted = df.sort_values([parent_key]).reset_index(drop=True)
        else:
            df_sorted = df.sort_values([parent_key, datetime_col]).reset_index(drop=True)
        
        n_groups = df_sorted[parent_key].nunique()
        
        valid_groups = 0
        for entity_id, group in df_sorted.groupby(parent_key):
            values = group[numeric_col].dropna()
            
            # 최소 길이 체크
            if len(values) < min_length or len(values) <= max_lag:
                continue
            
            if values.std() < 1e-10:
                continue
                
            valid_groups += 1
            
            try:
                # ACF 계산 (lag 0 제외)
                acf_values = acf(values, nlags=max_lag, fft=True)[1:]
                
                if np.any(np.isnan(acf_values)) or np.any(np.isinf(acf_values)):
                    continue
                    
                all_acfs.append(acf_values)
            except Exception as e:
                print(f"  [ACF Debug] ACF Computation Failed for entity {entity_id}: {e}")
                continue
                
        if not all_acfs:
            print(f"  [ACF Debug] {numeric_col}: No Valid ACF")
            return None
        
        mean_acf = np.mean(all_acfs, axis=0)
        
        if np.any(np.isnan(mean_acf)) or np.any(np.isinf(mean_acf)):
            return None
            
        # print(f"  [ACF Debug] {numeric_col}: 평균 ACF 계산 성공 (shape: {mean_acf.shape})")
        return mean_acf
    
    # # 메인 로직
    # print(f"\n[ACF] parent_key: {parent_key}, max_lag: {max_lag}, min_length: {min_length}")
    # print(f"[ACF] real_df shape: {real_df.shape}, syn_df shape: {syn_df.shape}")
    # print(f"[ACF] numerical features: {features}")
    
    results = {
        "per_feature": {},
        "acf_mae_overall": np.nan,
        "acf_max_diff_overall": np.nan,
    }
    
    mae_scores = []
    max_diff_scores = []
    
    for feat in features:
        if feat not in syn_df.columns:
            continue
        
        print(f"\n[ACF] Processing feature: {feat}")
        
        # 실제/합성 데이터의 평균 ACF 계산
        real_acf = _calculate_acf_per_entity(
            real_df, parent_key, time_column, feat, max_lag, min_length
        )
        syn_acf = _calculate_acf_per_entity(
            syn_df, parent_key, time_column, feat, max_lag, min_length
        )
        
        if real_acf is None or syn_acf is None:
            print(f"[ACF] {feat}: ACF 계산 실패 (real={real_acf is not None}, syn={syn_acf is not None})")
            continue
        
        # ACF 패턴 차이 계산
        acf_diff = np.abs(real_acf - syn_acf)
        
        # nan 체크 ⭐ 추가
        if np.any(np.isnan(acf_diff)) or np.any(np.isinf(acf_diff)):
            print(f"[ACF] {feat}: ACF 차이 계산에서 nan/inf 발생!")
            print(f"  real_acf: {real_acf}")
            print(f"  syn_acf: {syn_acf}")
            continue
        
        mae = float(np.mean(acf_diff))
        max_diff = float(np.max(acf_diff))
        
        # 최종 nan 체크 ⭐ 추가
        if np.isnan(mae) or np.isnan(max_diff):
            print(f"[ACF] {feat}: 최종 메트릭이 nan!")
            continue
        
        print(f"[ACF] {feat}: MAE={mae:.4f}, Max Diff={max_diff:.4f}")
        
        results["per_feature"][feat] = {
            "acf_mae": mae,
            "acf_max_diff": max_diff,
            "real_acf": real_acf.tolist(),
            "syn_acf": syn_acf.tolist(),
        }
        
        mae_scores.append(mae)
        max_diff_scores.append(max_diff)
    
    if mae_scores:
        results["acf_mae_overall"] = float(np.mean(mae_scores))
        results["acf_max_diff_overall"] = float(np.mean(max_diff_scores))
        print(f"\n[ACF] Overall - MAE: {results['acf_mae_overall']:.4f}, Max Diff: {results['acf_max_diff_overall']:.4f}")
    else:
        print(f"\n[ACF] Overall - 계산된 메트릭 없음!")
    
    return results