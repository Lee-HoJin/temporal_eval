import os
from os.path import join
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon as JSD
from scipy.stats import ks_2samp
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import utils

BASE_PATH = Path('/home/yjung/syntherela/experiments/data')
SYNTHETIC_DATA_PATH = BASE_PATH / 'synthetic'
REAL_DATA_PATH = BASE_PATH / 'original'

DATASETS = [# 'airbnb-simplified_subsampled'
            # 'rossmann_subsampled',
            'walmart_subsampled'
            # 'berka'
            # 'freddiemac'
]
# DATASETS = ['airbnb-simplified_subsampled', 'rossmann_subsampled', 'walmart_subsampled']
# MODELS = ['CLAVADDPM', 'RCTGAN', 'REALTABFORMER', 'RGCLD', 'SDV', 'TabDiT', 'ours']
MODELS = ['CLAVADDPM', 'RCTGAN', 'REALTABFORMER', 'RGCLD', 'SDV', 'RelDiff']

results = {}
ks_results = {}
lag1_diff_results = {}


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

def calculate_ks_test(real_values, syn_values):
    """두 분포 간의 KS 테스트를 수행합니다."""
    # NaN 값 제거
    real_clean = real_values.dropna()
    syn_clean = syn_values.dropna()
    
    if len(real_clean) == 0 or len(syn_clean) == 0:
        return np.nan, np.nan
    
    # KS 테스트
    ks_stat, p_value = ks_2samp(real_clean, syn_clean)
    return ks_stat, p_value

def calculate_lag1_differences(df, parent_key, datetime_col, numeric_col):
    """각 개체별로 lag-1 차분을 계산합니다. datetime이 없으면 PKey로 정렬"""
    all_diffs = []
    
    # 빈 데이터프레임 체크
    if df.empty:
        print(f"Warning: 빈 데이터프레임이 전달됨 - {numeric_col}")
        return np.array([])
    
    # 컬럼 존재 체크
    if numeric_col not in df.columns:
        print(f"Warning: 컬럼 '{numeric_col}'이 존재하지 않음")
        return np.array([])
    
    # parent_key 존재 체크  
    if parent_key not in df.columns:
        print(f"Warning: Parent key '{parent_key}'이 존재하지 않음")
        return np.array([])
    
    if datetime_col is None:
        # datetime 컬럼이 없는 경우: parent_key로만 정렬하고 연속된 행 간의 차이를 계산
        print(f"datetime 컬럼이 없음. {parent_key}로 정렬하여 연속 행 차이 계산")
        df_sorted = df.sort_values([parent_key]).reset_index(drop=True)
        
        for _, group in df_sorted.groupby(parent_key):
            if len(group) < 2:
                continue
            
            # 연속된 행들의 값 차이 계산
            values = group[numeric_col].dropna()
            if len(values) < 2:
                continue
                
            # lag-1 차분 계산 (연속된 행 간의 차이)
            diffs = values.diff().dropna()
            all_diffs.extend(diffs.tolist())
    else:
        # datetime 컬럼이 있는 경우: 기존 방식 (fkey, date 정렬)
        print(f"{parent_key}, {datetime_col}로 정렬하여 시계열 차이 계산")
        
        # datetime 컬럼 존재 체크
        if datetime_col not in df.columns:
            print(f"Warning: Datetime 컬럼 '{datetime_col}'이 존재하지 않음")
            return np.array([])
            
        df_sorted = df.sort_values([parent_key, datetime_col]).reset_index(drop=True)
        
        for _, group in df_sorted.groupby(parent_key):
            if len(group) < 2:
                continue
            
            # 날짜 순으로 정렬된 값들의 차이 계산
            values = group[numeric_col].dropna()
            if len(values) < 2:
                continue
                
            # lag-1 차분 계산
            diffs = values.diff().dropna()
            all_diffs.extend(diffs.tolist())
    
    if len(all_diffs) == 0:
        print(f"Warning: {numeric_col}에 대한 lag-1 차분이 계산되지 않음")
    
    return np.array(all_diffs)

def calculate_lag1_diff_metrics(real_diffs, syn_diffs, n_bins=20):
    """lag-1 차분 분포에 대한 KS 테스트와 JSD를 계산합니다."""
    if len(real_diffs) == 0 or len(syn_diffs) == 0:
        return np.nan, np.nan, np.nan
    
    # 1. KS 테스트
    ks_stat, p_value = ks_2samp(real_diffs, syn_diffs)
    
    # 2. JSD 계산을 위한 히스토그램 생성
    # 공통 범위 설정
    min_val = min(real_diffs.min(), syn_diffs.min())
    max_val = max(real_diffs.max(), syn_diffs.max())
    
    # 범위가 너무 작으면 약간 확장
    if abs(max_val - min_val) < 1e-10:
        min_val -= 0.1
        max_val += 0.1
    
    # 히스토그램 생성
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    real_hist, _ = np.histogram(real_diffs, bins=bins, density=True)
    syn_hist, _ = np.histogram(syn_diffs, bins=bins, density=True)
    
    # 정규화 (확률 분포로 변환)
    real_hist = real_hist / real_hist.sum() if real_hist.sum() > 0 else real_hist
    syn_hist = syn_hist / syn_hist.sum() if syn_hist.sum() > 0 else syn_hist
    
    # 작은 값 추가 (0으로 나누기 방지)
    eps = 1e-10
    real_hist = real_hist + eps
    syn_hist = syn_hist + eps
    
    # 재정규화
    real_hist = real_hist / real_hist.sum()
    syn_hist = syn_hist / syn_hist.sum()
    
    # JSD 계산
    jsd = JSD(real_hist, syn_hist)
    
    return ks_stat, p_value, jsd

def identify_numeric_columns(metadata, child_table_name):
    """수치형 컬럼을 식별합니다."""
    child_table_meta = metadata['tables'][child_table_name]
    numeric_cols = []
    
    for col, info in child_table_meta['columns'].items():
        if info['sdtype'] in ['numerical', 'float', 'int']:
            numeric_cols.append(col)
    
    return numeric_cols

# 메인 분석 루프
for dataset in DATASETS:
    print(f"\n============== 데이터셋 분석 시작: {dataset} ==============")
    metadata = utils.load_metadata(REAL_DATA_PATH, dataset)
    
    for relationship in metadata['relationships']:
        parent_table = relationship['parent_table_name']
        child_table = relationship['child_table_name']
        
        print(f"\n--- 테이블 관계: {parent_table} -> {child_table} ---")

        # 1. 실제 데이터
        real_data_path = REAL_DATA_PATH / dataset
        real_processed = utils.load_and_preprocess_data(real_data_path, metadata, parent_table, child_table)
        
        if real_processed is None:
            print(f"Warning: {dataset}의 {parent_table}->{child_table} 실제 데이터를 처리할 수 없습니다. 스킵합니다.")
            continue
        
        real_df, parent_key, datetime_col = real_processed
        child_table_meta = metadata['tables'][child_table]
        id_cols = {col for col, info in child_table_meta['columns'].items() if info['sdtype'] == 'id'}
        id_cols.add(relationship['child_foreign_key'])
        all_cols = set(child_table_meta['columns'].keys())
        if datetime_col:
            id_cols.add(datetime_col)
        cols_to_analyze = sorted(list(all_cols - id_cols))
        
        # 수치형 컬럼 식별
        numeric_cols = identify_numeric_columns(metadata, child_table)
        numeric_cols_to_analyze = [col for col in cols_to_analyze if col in numeric_cols]
        
        print(f"전체 분석 대상 컬럼: {cols_to_analyze}")
        print(f"수치형 분석 대상 컬럼: {numeric_cols_to_analyze}")
        
        # 기존 전이 행렬 계산
        for tm_col in cols_to_analyze:
            if dataset == "rossmann_subsampled":
                n_bins = 7
            else:
                n_bins = 5
            real_tm = calculate_transition_matrix(real_df, parent_key, tm_col, datetime_col, n_bins=n_bins)
            results[(dataset, child_table, 'Real', tm_col)] = real_tm
        
        # 수치형 컬럼에 대한 추가 분석
        for num_col in numeric_cols_to_analyze:
            # 실제 데이터의 lag-1 차분 계산
            real_lag1_diffs = calculate_lag1_differences(real_df, parent_key, datetime_col, num_col)
            lag1_diff_results[(dataset, child_table, 'Real', num_col)] = real_lag1_diffs
            
        # 2. 합성 데이터
        for model in MODELS:
            print(f"모델 분석 중: {model}")
            synthetic_data_path = SYNTHETIC_DATA_PATH / dataset / model / '1' / 'sample1'
            print(synthetic_data_path)
            synth_processed = utils.load_and_preprocess_data(synthetic_data_path, metadata, parent_table, child_table)
            
            if synth_processed is None:
                print(f"Warning: {model} 합성 데이터를 처리할 수 없습니다.")
                continue
                
            synth_df, _, _ = synth_processed
            
            # 기존 전이 행렬 계산
            for tm_col in cols_to_analyze:
                if tm_col in synth_df.columns:
                    if dataset == "rossmann_subsampled":
                        n_bins = 7
                    else:
                        n_bins = 5
                    synth_tm = calculate_transition_matrix(synth_df, parent_key, tm_col, datetime_col, n_bins=n_bins)
                    results[(dataset, child_table, model, tm_col)] = synth_tm
            
            # 수치형 컬럼에 대한 추가 분석
            for num_col in numeric_cols_to_analyze:
                if num_col in synth_df.columns:
                    # 원본 분포에 대한 KS 테스트 (기존 유지)
                    real_values = real_df[num_col]
                    syn_values = synth_df[num_col]
                    ks_stat, p_value = calculate_ks_test(real_values, syn_values)
                    ks_results[(dataset, child_table, model, num_col)] = {'ks_stat': ks_stat, 'p_value': p_value}
                    
                    # lag-1 차분 계산
                    syn_lag1_diffs = calculate_lag1_differences(synth_df, parent_key, datetime_col, num_col)
                    lag1_diff_results[(dataset, child_table, model, num_col)] = syn_lag1_diffs

    print("="*50)

# 기존 전이 행렬 분석 함수들
def transition_diff_L1(real_tm, syn_tm):
    real_tm, syn_tm = real_tm.fillna(0), syn_tm.fillna(0).reindex_like(real_tm)
    return np.mean(np.abs(real_tm.values - syn_tm.values))

def transition_diff_JSD(real_tm, syn_tm):
    real_tm, syn_tm = real_tm.fillna(0), syn_tm.fillna(0).reindex_like(real_tm)
    
    eps = 1e-10
    real_vals = real_tm.values + eps
    syn_vals = syn_tm.values + eps
    
    real_norm = real_vals / real_vals.sum(axis=1, keepdims=True)
    syn_norm = syn_vals / syn_vals.sum(axis=1, keepdims=True)
    
    jsd_values = []
    for i in range(real_norm.shape[0]):
        jsd = JSD(real_norm[i], syn_norm[i])
        jsd_values.append(jsd)
    
    return np.mean(jsd_values)

# 기존 전이 행렬 결과 처리
results_diff = []
for (dataset, child_table, column) in sorted(set((k[0], k[1], k[3]) for k in results.keys())):
    real_key = (dataset, child_table, 'Real', column)
    if real_key not in results or results[real_key].empty:
        continue
    real_tm = results[real_key]
        
    for model in MODELS:
        model_key = (dataset, child_table, model, column)
        if model_key not in results or results[model_key].empty:
            continue
        syn_tm = results[model_key]
        
        diff_l1 = transition_diff_L1(real_tm, syn_tm)
        diff_jsd = transition_diff_JSD(real_tm, syn_tm)
        
        results_diff.append({
            'dataset': dataset,
            'table': child_table,
            'column': column,
            'model': model,
            'l1': diff_l1,
            'jsd': diff_jsd,
        })

# KS 테스트 결과 처리
ks_results_list = []
for (dataset, child_table, model, column) in ks_results.keys():
    result = ks_results[(dataset, child_table, model, column)]
    ks_results_list.append({
        'dataset': dataset,
        'table': child_table,
        'column': column,
        'model': model,
        'ks_stat': result['ks_stat'],
        'p_value': result['p_value']
    })

# lag-1 차분 KS 테스트 및 JSD 결과 처리
lag1_diff_metrics_results = []
for (dataset, child_table, column) in sorted(set((k[0], k[1], k[3]) for k in lag1_diff_results.keys() if k[2] == 'Real')):
    real_key = (dataset, child_table, 'Real', column)
    if real_key not in lag1_diff_results:
        continue
    real_diffs = lag1_diff_results[real_key]
    
    for model in MODELS:
        model_key = (dataset, child_table, model, column)
        if model_key not in lag1_diff_results:
            continue
        syn_diffs = lag1_diff_results[model_key]
        
        # lag-1 차분 분포에 대한 KS 테스트 및 JSD 계산
        ks_stat_lag1, p_value_lag1, jsd_lag1 = calculate_lag1_diff_metrics(real_diffs, syn_diffs)
        
        lag1_diff_metrics_results.append({
            'dataset': dataset,
            'table': child_table,
            'column': column,
            'model': model,
            'ks_stat_lag1_diff': ks_stat_lag1,
            'p_value_lag1_diff': p_value_lag1,
            'jsd_lag1_diff': jsd_lag1
        })

# 결과 DataFrame 생성
dataset_name = DATASETS[0]
RESULTS_DIR = Path('/home/yjung/syntherela/experiments/evaluation_linked/hojin/results/')
df_diff = pd.DataFrame(results_diff)
df_ks = pd.DataFrame(ks_results_list)
df_lag1_metrics = pd.DataFrame(lag1_diff_metrics_results)

# print("=== 전이 행렬 기반 결과 (기존) ===")
# print("df_diff\n", df_diff)
# df_diff.to_csv(join(RESULTS_DIR, f"{dataset_name}_temporal_results_tm_based.csv"))

# print("\n=== 원본 분포 KS 테스트 결과 ===")
# print("df_ks\n", df_ks)
# df_ks.to_csv(join(RESULTS_DIR, f"{dataset_name}_temporal_results_KST.csv"))

# print("\n=== Lag-1 차분 분포: KS 테스트 및 JSD 결과 ===")
# print("df_lag1_metrics\n", df_lag1_metrics)
# df_lag1_metrics.to_csv(join(RESULTS_DIR, f"{dataset_name}_temporal_results_lag1dff.csv"))

# 요약 통계 생성 함수
def create_summary(df, metric_col, metric_name):
    summary = (
        df.groupby('model')[metric_col]
        .agg(['mean', 'std', 'count'])
        .assign(se=lambda x: x['std'] / np.sqrt(x['count']))
        .reset_index()
    )
    summary['mean±se'] = summary.apply(
        lambda x: f"{x['mean']:.4f} ± {x['se']:.4f}", axis=1
    )
    return summary[['model', 'mean±se']].rename(columns={'mean±se': metric_name})

# 기존 L1, JSD 요약
df_summary_l1 = create_summary(df_diff, 'l1', 'L1')
df_summary_jsd = create_summary(df_diff, 'jsd', 'JSD')

# 새로운 메트릭 요약
if not df_ks.empty:
    df_summary_ks = create_summary(df_ks, 'ks_stat', 'KS_stat_original')
else:
    df_summary_ks = pd.DataFrame(columns=['model', 'KS_stat_original'])

if not df_lag1_metrics.empty:
    df_summary_ks_lag1 = create_summary(df_lag1_metrics, 'ks_stat_lag1_diff', 'KS_stat_lag1_diff')
    df_summary_jsd_lag1 = create_summary(df_lag1_metrics, 'jsd_lag1_diff', 'JSD_lag1_diff')
else:
    df_summary_ks_lag1 = pd.DataFrame(columns=['model', 'KS_stat_lag1_diff'])
    df_summary_jsd_lag1 = pd.DataFrame(columns=['model', 'JSD_lag1_diff'])

# 통합 요약
df_summary_combined = df_summary_l1.copy()
if 'JSD' in df_summary_jsd.columns:
    df_summary_combined = df_summary_combined.merge(df_summary_jsd[['model', 'JSD']], on='model', how='left')
if 'KS_stat_original' in df_summary_ks.columns:
    df_summary_combined = df_summary_combined.merge(df_summary_ks[['model', 'KS_stat_original']], on='model', how='left')
if 'KS_stat_lag1_diff' in df_summary_ks_lag1.columns:
    df_summary_combined = df_summary_combined.merge(df_summary_ks_lag1[['model', 'KS_stat_lag1_diff']], on='model', how='left')
if 'JSD_lag1_diff' in df_summary_jsd_lag1.columns:
    df_summary_combined = df_summary_combined.merge(df_summary_jsd_lag1[['model', 'JSD_lag1_diff']], on='model', how='left')

print("\n=== 통합 요약 ===")
print(df_summary_combined)

# 개별 결과도 저장할 수 있도록 딕셔너리로 반환
evaluation_results = {
    'transition_matrix': df_diff,
    'ks_test_original': df_ks,
    'lag1_diff_metrics': df_lag1_metrics,
    'summary': df_summary_combined
}

df_summary_combined.to_csv(join(RESULTS_DIR, f"{dataset_name}_temporal_results_combined.csv"))

# print("\n=== 분석 완료 ===")
# print("사용 가능한 결과:")
# print("- evaluation_results['transition_matrix']: 전이 행렬 L1/JSD 결과")
# print("- evaluation_results['ks_test_original']: 원본 분포 KS 테스트 결과")
# print("- evaluation_results['lag1_diff_metrics']: Lag-1 차분 분포의 KS 테스트 및 JSD 결과")
# print("- evaluation_results['summary']: 모든 메트릭 요약")