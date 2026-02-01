"""
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
하고 실행할 것
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval_privacy import (
    IdentifiabilityScore, 
    kAnonymization,
    DeltaPresence,
    kMap,
    lDiversityDistinct,
)

#                 0           1             2           3        4        5 
syn_models = ['CLAVADDPM', 'RCTGAN', 'REALTABFORMER', 'RGCLD', 'SDV', 'RelDiff']
syn_model_name = syn_models[0]

# dataset_name = 'rossmann_subsampled'
# table_name = 'historical'

dataset_name = 'walmart_subsampled'
table_name = 'depts'
# table_name = 'features'

REAL_PATH = f"/home/yjung/syntherela/experiments/data/original/{dataset_name}/{table_name}.csv"
SYN_PATH  = f"/home/yjung/syntherela/experiments/data/synthetic/{dataset_name}/{syn_model_name}/1/sample1/{table_name}.csv"

real_df = pd.read_csv(REAL_PATH)
syn_df  = pd.read_csv(SYN_PATH)

print(f"Original Real data shape: {real_df.shape}")
print(f"Original Synthetic data shape: {syn_df.shape}")

# NaN 확인
print(f"\nNaN 개수 (Real): {real_df.isnull().sum().sum()}")
print(f"NaN 개수 (Synthetic): {syn_df.isnull().sum().sum()}")

# =====================================
# Categorical 인코딩 + NaN 처리
# =====================================
def encode_and_impute(real_df, syn_df):
    """
    1. Categorical 컬럼 인코딩
    2. NaN 값 처리 (imputation)
    """
    real_encoded = real_df.copy()
    syn_encoded = syn_df.copy()
    
    label_encoders = {}
    
    # 1. Categorical 컬럼 인코딩
    cat_cols = real_df.select_dtypes(include=['object', 'category']).columns
    print(f"\n인코딩할 categorical 컬럼 ({len(cat_cols)}개): {list(cat_cols)}")
    
    for col in cat_cols:
        le = LabelEncoder()
        
        # Real 데이터 인코딩 (NaN을 특수 카테고리로 처리)
        real_encoded[col] = real_encoded[col].astype(str).fillna('__MISSING__')
        le.fit(real_encoded[col])
        real_encoded[col] = le.transform(real_encoded[col])
        
        # Synthetic 데이터 인코딩
        if col in syn_encoded.columns:
            syn_encoded[col] = syn_encoded[col].astype(str).fillna('__MISSING__')
            
            # Synthetic에 Real에 없는 값이 있으면 처리
            syn_values = syn_encoded[col].values
            unknown_mask = ~np.isin(syn_values, le.classes_)
            
            if unknown_mask.any():
                n_unknown = unknown_mask.sum()
                if n_unknown < 100:  # 적은 경우만 경고
                    print(f"  경고: '{col}' - real에 없는 값 {n_unknown}개 -> 첫 번째 카테고리로 대체")
                syn_values[unknown_mask] = le.classes_[0]
            
            syn_encoded[col] = le.transform(syn_values)
        
        label_encoders[col] = le
        print(f"  ✓ '{col}' 완료: {len(le.classes_)} unique values")
    
    # 2. 수치형 컬럼 NaN 처리 (median imputation)
    numeric_cols = real_encoded.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) > 0:
        print(f"\nNaN 처리할 수치형 컬럼 ({len(numeric_cols)}개)")
        
        for col in numeric_cols:
            real_nan_count = real_encoded[col].isnull().sum()
            syn_nan_count = syn_encoded[col].isnull().sum() if col in syn_encoded.columns else 0
            
            if real_nan_count > 0 or syn_nan_count > 0:
                imputer = SimpleImputer(strategy='median')
                real_encoded[col] = imputer.fit_transform(real_encoded[[col]]).ravel()
                
                if col in syn_encoded.columns:
                    syn_encoded[col] = imputer.transform(syn_encoded[[col]]).ravel()
                
                print(f"  ✓ '{col}': real NaN={real_nan_count}, syn NaN={syn_nan_count} -> median={imputer.statistics_[0]:.2f}")
    
    print(f"\n최종 데이터 확인:")
    print(f"  Real: {real_encoded.shape}, NaN={real_encoded.isnull().sum().sum()}")
    print(f"  Synthetic: {syn_encoded.shape}, NaN={syn_encoded.isnull().sum().sum()}")
    
    return real_encoded, syn_encoded, label_encoders

# 인코딩 및 Imputation 수행
real_encoded, syn_encoded, encoders = encode_and_impute(real_df, syn_df)

# =====================================
# DataLoader 생성
# =====================================

if dataset_name == 'rossmann_subsampled':
    if table_name == 'historical':
        sensitive_cols = ['Customers']
    elif table_name == 'store':
        sensitive_cols = ['CompetitionDistance']

elif dataset_name == 'walmart_subsampled':
    if table_name == 'depts':
        sensitive_cols = ['Weekly_Sales']
    elif table_name == 'features':
        sensitive_cols = ['Unemployement']

real_loader = GenericDataLoader(real_encoded, sensitive_features = sensitive_cols)
syn_loader = GenericDataLoader(syn_encoded, sensitive_features = sensitive_cols)


# Privacy 메트릭 계산
# =====================================
print("\n" + "=" * 60)
print("Privacy 메트릭 계산")
print("=" * 60)

print("사용된 Synthetic 모델:", syn_model_name)

privacy_results = {}

# 1. Identifiability Score
print("\n1. Identifiability Score 계산 중...")
try:
    id_metric = IdentifiabilityScore()
    id_score = id_metric.evaluate(real_loader, syn_loader)
    privacy_results['identifiability_score'] = id_score
    print(f"   ✓ Identifiability Score: {id_score}")  # dict일 수 있으므로 포맷 제거
except Exception as e:
    print(f"   ✗ 에러: {e}")

# 2. k-Anonymization
print("\n2. k-Anonymization 계산 중...")
try:
    kanon_metric = kAnonymization()
    kanon_result = kanon_metric.evaluate(real_loader, syn_loader)
    privacy_results['k_anonymization'] = kanon_result
    print(f"   ✓ k-Anonymization: {kanon_result}")
except Exception as e:
    print(f"   ✗ 에러: {e}")

# 3. l-Diversity
print("\n3. l-Diversity 계산 중...")
try:
    ldiv_metric = lDiversityDistinct()
    ldiv_result = ldiv_metric.evaluate(real_loader, syn_loader)
    privacy_results['l_diversity'] = ldiv_result
    print(f"   ✓ l-Diversity: {ldiv_result}")
except Exception as e:
    print(f"   ✗ 에러: {e}")
    
# 4. Delta Presence
print("\n4. Delta Presence 계산 중...")
try:
    delta_metric = DeltaPresence()
    delta_score = delta_metric.evaluate(real_loader, syn_loader)
    privacy_results['delta_presence'] = delta_score
    print(f"   ✓ Delta Presence: {delta_score}")  # dict일 수 있으므로 포맷 제거
except Exception as e:
    print(f"   ✗ 에러: {e}")
    
# 5. k-Map
print("\n5. k-Map 계산 중...")
try:
    kmap_metric = kMap()
    kmap_score = kmap_metric.evaluate(real_loader, syn_loader)
    privacy_results['kmap'] = kmap_score
    print(f"   ✓ k-Map: {kmap_score}")
    
    # sensitive_features 확인
    print(f"Sensitive features: {real_loader.sensitive_features}")
    print(f"All features: {list(real_loader.columns)}")

    # quasi-identifiers 계산
    from synthcity.metrics.weighted_metrics import _utils
    quasi_identifiers = _utils.get_features(real_loader, real_loader.sensitive_features)
    print(f"Quasi-identifiers (kMap에서 사용): {quasi_identifiers}")
except Exception as e:
    print(f"   ✗ 에러: {e}")


"""
# =====================================
# 결과 요약 및 저장
# =====================================
print("\n" + "=" * 60)
print("Privacy 메트릭 최종 결과")
print("=" * 60)


if privacy_results:
    # 딕셔너리 flatten
    flattened_results = {}
    for k, v in privacy_results.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flattened_results[f"{k}.{sub_k}"] = sub_v
        else:
            flattened_results[k] = v
    
    # 보기 좋게 출력
    for metric_name, value in flattened_results.items():
        if isinstance(value, (int, float)):
            print(f"{metric_name:40s}: {value:.6f}" if isinstance(value, float) else f"{metric_name:40s}: {value}")
        else:
            print(f"{metric_name:40s}: {value}")
    
    # CSV로 저장
    results_df = pd.DataFrame([flattened_results])
    output_path = "privacy_metrics_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ 결과가 {output_path}에 저장되었습니다.")
    
    # 결과 DataFrame 미리보기
    print("\n저장된 결과:")
    print(results_df.T)
else:
    print("계산된 메트릭이 없습니다.")
"""