import pandas as pd
import numpy as np
from typing import Dict, List, Optional

import metadata
import utils

def make_lag_k_df(
    df: pd.DataFrame,
    features: List[str],
    lag: int = 1
) -> pd.DataFrame:
    """
    Create a DataFrame with lag-k features.
    
    Args:
        df: Original DataFrame.
        features: List of feature column names to create lagged versions for.
        lag: The lag value (default is 1).
    
    Returns:
        DataFrame with lag-k features.
    """
    lagged_df = df.copy()
    
    for feat in features:
        lagged_col_name = f"{feat}_lag_{lag}"
        lagged_df[lagged_col_name] = df[feat].shift(lag)
    
    return lagged_df

def lag_k_difference(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    features: List[str],
    lag: int = 1
) -> Dict:
    
    # real_df, syn_df = utils.scale_features(real_df, syn_df, features)
    
    results = {
            "per_feature": {},
            "lag_k_difference_overall": np.nan
        }

    diffs = []

    # Lag k difference calculation
    real_lagged = make_lag_k_df(real_df[features], features, lag)
    syn_lagged = make_lag_k_df(syn_df[features], features, lag)

    for feats in features:
        real_lagged = real_lagged[[f"{feats}_lag_{lag}"]]
        syn_lagged = syn_lagged[[f"{feats}_lag_{lag}"]]
        
        # Compute differences
        real_diff = real_df[feats] - real_lagged
        syn_diff = syn_df[feats] - syn_lagged

        # Compute absolute differences
        abs_diffs_series = np.abs(real_diff - syn_diff)
        abs_diffs = abs_diffs_series.mean(skipna=True)
        
        results['per_feature'][feats] = abs_diffs
        diffs.append(abs_diffs)
        
        print(f"feature: {feats} | abs_diffs: {abs_diffs}")

    if diffs:
        results["lag_k_difference_overall"] = np.mean(diffs)

    return results

## 고치는 중
def calculate_transition_matrix(df, parent_key, state_col, datetime_col=None, n_bins=5):
    df_tm = df.copy()
    
    # 빈 데이터프레임 체크
    if df_tm.empty:
        print(f"Warning: 빈 데이터프레임이 전달됨 - {state_col}")
        return pd.DataFrame()
    
    # state_col이 존재하는지 체크
    if state_col not in df_tm.columns:
        print(f"Warning: 컬럼 '{state_col}'이 존재하지 않음")
        return pd.DataFrame()
    
    # 해당 컬럼의 유효한 값이 있는지 체크
    valid_values = df_tm[state_col].dropna()
    if len(valid_values) == 0:
        print(f"Warning: 컬럼 '{state_col}'에 유효한 값이 없음")
        return pd.DataFrame()
    
    # 상태 컬럼이 수치형이면 구간화
    if pd.api.types.is_numeric_dtype(df_tm[state_col]):
        # 빈 배열 체크 추가
        if len(valid_values) < 2:
            print(f"Warning: 수치형 컬럼 '{state_col}'의 유효한 값이 너무 적음 (< 2개)")
            return pd.DataFrame()
            
        try:
            df_tm['state'] = pd.cut(df_tm[state_col], bins=n_bins, labels=False, include_lowest=True)
            labels = range(n_bins)
        except ValueError as e:
            print(f"Warning: pd.cut 실패 - {state_col}: {e}")
            return pd.DataFrame()
    else: # 범주형이면 그대로 사용
        df_tm['state'] = df_tm[state_col].astype('category')
        labels = df_tm['state'].cat.categories
        if len(labels) == 0:
            print(f"Warning: 범주형 컬럼 '{state_col}'에 카테고리가 없음")
            return pd.DataFrame()
    
    transitions = []
    
    if datetime_col is None:
        # datetime 컬럼이 없는 경우: parent_key로만 정렬하고 연속된 행 간의 전이 계산
        df_tm = df_tm.sort_values([parent_key]).reset_index(drop=True)
        
        for _, group in df_tm.groupby(parent_key):
            if len(group) < 2:
                continue
            # 현재 상태와 다음 상태(shift)를 묶어 전이 쌍 생성
            current_states = group['state'][:-1]
            next_states = group['state'][1:]
            transitions.extend(zip(current_states, next_states))
    else:
        # datetime 컬럼이 있는 경우: 기존 방식
        df_tm = df_tm.sort_values([parent_key, datetime_col]).reset_index(drop=True)
        
        for _, group in df_tm.groupby(parent_key):
            if len(group) < 2:
                continue
            # 현재 상태와 다음 상태(shift)를 묶어 전이 쌍 생성
            current_states = group['state'][:-1]
            next_states = group['state'][1:]
            transitions.extend(zip(current_states, next_states))

    if not transitions:
        print(f"Warning: 전이 쌍이 생성되지 않음 - {state_col}")
        return pd.DataFrame()

    # 모든 전이 쌍으로 전이 행렬 계산
    try:
        counts = pd.crosstab(
            pd.Series([t[0] for t in transitions], name='current_state'),
            pd.Series([t[1] for t in transitions], name='next_state')
        ).reindex(index=labels, columns=labels, fill_value=0)
        
        # 행의 합으로 나누어 확률로 정규화
        tm = counts.div(counts.sum(axis=1), axis=0).fillna(0)
        return tm
    except Exception as e:
        print(f"Warning: 전이 행렬 계산 실패 - {state_col}: {e}")
        return pd.DataFrame()