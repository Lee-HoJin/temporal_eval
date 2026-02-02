import pandas as pd
import numpy as np
import json

from metadata import Metadata
from pathlib import Path
import os
from typing import Dict, List, Optional

from sklearn.preprocessing import MinMaxScaler

def load_metadata(real_data_path, dataset_name):
    """지정된 데이터셋의 메타데이터 파일을 불러옵니다."""
    metadata_path = os.path.join(real_data_path, dataset_name, 'metadata.json')
    with open(metadata_path, 'r') as f:
        return json.load(f)

def scale_features(real_df, synth_df, features):
    real_df = real_df.copy()
    synth_df = synth_df.copy()

    scaler = MinMaxScaler()
    scaler.fit(real_df[features])

    real_df[features] = scaler.transform(real_df[features])
    synth_df[features] = scaler.transform(synth_df[features])
    
    return real_df, synth_df

def get_fk(real_data_path, table_name):
    metadata = Metadata().load_from_json(Path(real_data_path) / f"metadata.json")
    
    fk = metadata.get_primary_key(table_name)
    
    return fk

def get_frequencies(real_data, synthetic_data):
    """
    Get normalized frequency distributions for real and synthetic data.
    
    Args:
        real_data: array-like
        synthetic_data: array-like
    
    Returns:
        f_real: normalized frequencies (probabilities) for real data
        f_syn: normalized frequencies (probabilities) for synthetic data
    """
    # Get all unique categories
    all_categories = sorted(set(real_data) | set(synthetic_data))
    
    # Count frequencies
    real_counts = pd.Series(real_data).value_counts()
    syn_counts = pd.Series(synthetic_data).value_counts()
    
    # Align to all categories (fill missing with 0)
    f_real = np.array([real_counts.get(cat, 0) for cat in all_categories])
    f_syn = np.array([syn_counts.get(cat, 0) for cat in all_categories])
    
    # Normalize to probabilities
    f_real = f_real / (f_real.sum() + 1e-12)
    f_syn = f_syn / (f_syn.sum() + 1e-12)
    
    return f_real, f_syn

def get_datetime_col_info(metadata, table_name):
    """메타데이터에서 날짜/시간 컬럼의 이름과 포맷을 찾습니다."""
    for col, info in metadata['tables'][table_name]['columns'].items():
        if info['sdtype'] == 'datetime':
            return col, info.get('datetime_format')
    return None, None

def load_and_preprocess_data(data_path, metadata, parent_table_name, child_table_name):
    """개선된 부모-자식 테이블 병합 (ROBUST VERSION)"""
    
    # 1. 관계 정보 추출
    relationship = next(
        (r for r in metadata['relationships'] 
         if r['parent_table_name'] == parent_table_name 
         and r['child_table_name'] == child_table_name),
        None
    )
    
    if relationship is None:
        print(f"Error: {parent_table_name}-{child_table_name} 관계를 찾을 수 없음")
        return None
    
    parent_key = relationship['parent_primary_key']
    child_key = relationship['child_foreign_key']
    
    # 2. 데이터 로드
    try:
        parent_df = pd.read_csv(os.path.join(data_path, f"{parent_table_name}.csv"))
        child_df = pd.read_csv(os.path.join(data_path, f"{child_table_name}.csv"))
    except FileNotFoundError as e:
        print(f"Error: 파일을 찾을 수 없음 - {e}")
        return None
    
    # 3. 필수 컬럼 존재 확인
    if parent_key not in parent_df.columns:
        print(f"Error: Parent key '{parent_key}' not found in {parent_table_name}")
        return None
    
    if child_key not in child_df.columns:
        print(f"Error: Child key '{child_key}' not found in {child_table_name}")
        return None
    
    # 4. 키 타입 통일 (SIMPLIFIED)
    parent_df, child_df = unify_key_types(
        parent_df, child_df, parent_key, child_key
    )
    
    # 5. 병합 전 진단
    diagnosis = diagnose_merge(parent_df, child_df, parent_key, child_key)
    
    if not diagnosis['can_merge']:
        print(f"Error: 병합 불가능 - {diagnosis['reason']}")
        return None
    
    if diagnosis['warnings']:
        for warning in diagnosis['warnings']:
            print(f"Warning: {warning}")
    
    # 6. 병합 수행
    try:
        merged_df = pd.merge(
            child_df, 
            parent_df, 
            left_on=child_key, 
            right_on=parent_key, 
            how='inner',
            validate='many_to_one'  # 👈 관계 검증 추가!
        )
    except pd.errors.MergeError as e:
        print(f"Error: 병합 실패 - {e}")
        return None
    
    if merged_df.empty:
        print("Error: 병합 결과가 비어있음")
        return None
    
    print(f"✅ Merge Completed: {len(merged_df)} rows")
    
    # 7. Datetime 컬럼 처리
    datetime_col, datetime_format = get_datetime_col_info(metadata, child_table_name)
    
    if datetime_col and datetime_col in merged_df.columns:
        try:
            merged_df[datetime_col] = pd.to_datetime(
                merged_df[datetime_col], 
                # format=datetime_format,
                errors='coerce'
            ).dt.floor('D')
            
            # NaT가 너무 많으면 경고
            nat_ratio = merged_df[datetime_col].isna().sum() / len(merged_df)
            if nat_ratio > 0.1:
                print(f"Warning: {nat_ratio:.1%}의 날짜 변환 실패")
                
        except Exception as e:
            print(f"Warning: datetime 변환 실패 ({e}), datetime_col을 None으로 설정")
            datetime_col = None
    
    # 8. 정렬
    sort_cols = [parent_key]
    if datetime_col and datetime_col in merged_df.columns:
        sort_cols.append(datetime_col)
    
    merged_df = merged_df.sort_values(by=sort_cols).reset_index(drop=True)
    
    return merged_df, parent_key, datetime_col


def unify_key_types(parent_df, child_df, parent_key, child_key):
    """키 타입 통일 (SIMPLIFIED)"""
    
    # 원본 백업
    parent_original = parent_df[parent_key].copy()
    child_original = child_df[child_key].copy()
    
    # 전략 1: 숫자로 변환 시도
    parent_numeric = pd.to_numeric(parent_df[parent_key], errors='coerce')
    child_numeric = pd.to_numeric(child_df[child_key], errors='coerce')
    
    parent_success = (parent_numeric.notna().sum() / len(parent_numeric)) > 0.9
    child_success = (child_numeric.notna().sum() / len(child_numeric)) > 0.9
    
    if parent_success and child_success:
        parent_df[parent_key] = parent_numeric
        child_df[child_key] = child_numeric
        # print(f"✅ 키를 숫자로 변환: {parent_key}, {child_key}")
        return parent_df, child_df
    
    # 전략 2: 문자열로 변환
    parent_df[parent_key] = parent_original.astype(str).str.strip()
    child_df[child_key] = child_original.astype(str).str.strip()
    # print(f"✅ 키를 문자열로 변환: {parent_key}, {child_key}")
    
    return parent_df, child_df


def diagnose_merge(parent_df, child_df, parent_key, child_key):
    """병합 가능 여부 진단 (NO HEURISTIC FALLBACK!)"""
    
    result = {
        'can_merge': False,
        'reason': '',
        'warnings': []
    }
    
    # NaN 체크
    parent_valid = parent_df[parent_key].dropna()
    child_valid = child_df[child_key].dropna()
    
    parent_nan_ratio = (len(parent_df) - len(parent_valid)) / len(parent_df)
    child_nan_ratio = (len(child_df) - len(child_valid)) / len(child_df)
    
    if parent_nan_ratio > 0.1:
        result['warnings'].append(
            f"Parent key에 {parent_nan_ratio:.1%} NaN (병합 시 제외됨)"
        )
    
    if child_nan_ratio > 0.1:
        result['warnings'].append(
            f"Child key에 {child_nan_ratio:.1%} NaN (병합 시 제외됨)"
        )
    
    if len(parent_valid) == 0 or len(child_valid) == 0:
        result['reason'] = "유효한 키가 없음"
        return result
    
    # 공통 키 확인
    parent_keys = set(parent_valid)
    child_keys = set(child_valid)
    common_keys = parent_keys.intersection(child_keys)
    
    match_ratio = len(common_keys) / len(child_keys)
    
    # print(f"📊 Parent 고유 키: {len(parent_keys)}")
    # print(f"📊 Child 고유 키: {len(child_keys)}")
    # print(f"📊 공통 키: {len(common_keys)} ({match_ratio:.1%})")
    
    # ✅ 공통 키가 없으면 병합 불가 (NO FALLBACK!)
    if len(common_keys) == 0:
        result['reason'] = "공통 키가 없음 - FK 관계가 손상됨"
        print(f"Parent 키 샘플: {list(parent_keys)[:5]}")
        print(f"Child 키 샘플: {list(child_keys)[:5]}")
        return result
    
    # 매칭률이 너무 낮으면 경고
    if match_ratio < 0.5:
        result['warnings'].append(
            f"Child의 {match_ratio:.1%}만 Parent와 매칭됨 "
            f"({len(child_keys) - len(common_keys)}개 고아 레코드 제외됨)"
        )
    
    # Cardinality 체크 (Cartesian product 방지)
    sample_common = list(common_keys)[:10]
    total_expected = 0
    
    for key in sample_common:
        parent_count = (parent_valid == key).sum()
        child_count = (child_valid == key).sum()
        total_expected += parent_count * child_count
    
    avg_per_key = total_expected / len(sample_common)
    estimated_total = avg_per_key * len(common_keys)
    
    if estimated_total > 10_000_000:  # 1000만 행 이상
        result['reason'] = f"예상 병합 크기가 너무 큼 ({estimated_total:.0f} rows)"
        return result
    
    if estimated_total > 1_000_000:  # 100만 행 이상
        result['warnings'].append(
            f"예상 병합 크기: {estimated_total:.0f} rows (시간이 걸릴 수 있음)"
        )
    
    result['can_merge'] = True
    return result


def debug_temporal_overlap(real_df, synth_df, time_col='Date', target_col='IsHoliday'):
    print(f"🔍 디버깅: {target_col} 컬럼 및 시간 범위 확인")
    print("-" * 50)
    
    # 1. 날짜 형식 변환 및 범위 확인
    real_df[time_col] = pd.to_datetime(real_df[time_col])
    synth_df[time_col] = pd.to_datetime(synth_df[time_col])
    
    r_min, r_max = real_df[time_col].min(), real_df[time_col].max()
    s_min, s_max = synth_df[time_col].min(), synth_df[time_col].max()
    
    print(f"📅 Real Date Range : {r_min} ~ {r_max}")
    print(f"📅 Synth Date Range: {s_min} ~ {s_max}")
    
    # 2. 겹치는 기간 확인
    overlap_start = max(r_min, s_min)
    overlap_end = min(r_max, s_max)
    
    if overlap_start > overlap_end:
        print("❌ [CRITICAL] 겹치는 시간 구간이 전혀 없습니다! (JSD = 0의 원인)")
        return
    else:
        print(f"✅ Overlap Period  : {overlap_start} ~ {overlap_end}")

    # 3. 데이터 타입 확인
    print(f"\n🏷️ Data Types:")
    print(f"   Real [{target_col}]: {real_df[target_col].dtype} (예: {real_df[target_col].iloc[0]})")
    print(f"   Synth [{target_col}]: {synth_df[target_col].dtype} (예: {synth_df[target_col].iloc[0]})")
    
    if real_df[target_col].dtype != synth_df[target_col].dtype:
        print("⚠️ [WARNING] 데이터 타입이 다릅니다! (Bool vs Float 등)")
        print("   -> 비교 전 통일 필요 (예: .astype(str) or .astype(int))")

    # 4. 실제 구간별 데이터 존재 여부 샘플링 (첫 월/주)
    # 월단위 binning 예시
    sample_bin = real_df[time_col].dt.to_period('M').astype(str).unique()[0]
    
    r_count = real_df[real_df[time_col].dt.to_period('M').astype(str) == sample_bin].shape[0]
    s_count = synth_df[synth_df[time_col].dt.to_period('M').astype(str) == sample_bin].shape[0]
    
    print(f"\n📦 Sample Bin ({sample_bin}) Counts:")
    print(f"   Real: {r_count} rows")
    print(f"   Synth: {s_count} rows")
    
    if r_count > 0 and s_count == 0:
        print("❌ [CHECK] Real 데이터는 있는데 Synth 데이터가 해당 구간에 없습니다.")