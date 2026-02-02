import pandas as pd
import numpy as np

import metadata

def load_and_preprocess_data(data_path, metadata, parent_table_name, child_table_name):
    """부모-자식 테이블을 로드하고 병합 및 전처리합니다."""
    
    rels = metadata.get_relationships(parent_table_name)
    
    try:
        # 2. 리스트 안에서 현재 작업하려는 child_table_name과 일치하는 관계 하나를 찾음
        relationship = next(r for r in rels if r['child_table_name'] == child_table_name)
        
        # 3. 찾은 딕셔너리에서 키 추출
        parent_key = relationship['parent_primary_key']
        child_key = relationship['child_foreign_key']
        
        print(f"매핑 확인: {parent_table_name}({parent_key}) -> {child_table_name}({child_key})")
        
    except StopIteration:
        print(f"Error: {parent_table_name}와 {child_table_name} 사이의 관계를 찾을 수 없습니다.")
        return None
    
    # datetime_col을 미리 초기화 (참조 에러 방지)
    datetime_col = None
    
    try:
        parent_df = pd.read_csv(data_path + f"{parent_table_name}.csv")
        child_df = pd.read_csv(data_path + f"{child_table_name}.csv")
    except FileNotFoundError:
        print(f"Warning: 파일을 찾을 수 없습니다 - {data_path}")
        return None

    # 키가 실제로 존재하는지 확인
    if parent_key not in parent_df.columns:
        print(f"Error: Parent key '{parent_key}' not found in {parent_table_name}.csv")
        return None
    
    if child_key not in child_df.columns:
        print(f"Error: Child key '{child_key}' not found in {child_table_name}.csv")
        return None

    # 키 컬럼의 데이터 타입 통일 (더 안전한 방식)
    original_parent_key = parent_df[parent_key].copy()
    original_child_key = child_df[child_key].copy()
    
    # 먼저 숫자 변환 시도
    try:
        parent_numeric = pd.to_numeric(parent_df[parent_key], errors='coerce')
        child_numeric = pd.to_numeric(child_df[child_key], errors='coerce')
        
        # NaN이 너무 많이 생기면 숫자 변환 포기
        parent_nan_ratio = parent_numeric.isna().sum() / len(parent_numeric)
        child_nan_ratio = child_numeric.isna().sum() / len(child_numeric)
        
        if parent_nan_ratio > 0.1 or child_nan_ratio > 0.1:  # 10% 이상 NaN이면 포기
            print(f"숫자 변환 시 NaN 비율이 높음 (Parent: {parent_nan_ratio:.2%}, Child: {child_nan_ratio:.2%})")
            print("문자열 변환을 시도합니다.")
            
            # 문자열로 변환
            parent_df[parent_key] = original_parent_key.astype(str)
            child_df[child_key] = original_child_key.astype(str)
            print(f"키 컬럼을 문자열로 변환: {parent_key}, {child_key}")
        else:
            # 숫자 변환 성공
            parent_df[parent_key] = parent_numeric
            child_df[child_key] = child_numeric
            print(f"키 컬럼을 숫자로 변환: {parent_key}, {child_key}")
            
    except Exception as e:
        print(f"키 변환 실패, 원본 유지: {e}")
        # 변환 실패 시 원본 유지
        parent_df[parent_key] = original_parent_key
        child_df[child_key] = original_child_key
    
    print(f"변환 후 - Parent key '{parent_key}' dtype: {parent_df[parent_key].dtype}")
    
    # === 병합 전 진단 정보 ===
    print(f"\n=== 병합 전 진단 ===")
    print(f"Parent 테이블 크기: {len(parent_df)} 행")
    print(f"Child 테이블 크기: {len(child_df)} 행")
        
    # 유효한 값만으로 고유값 계산
    parent_valid = parent_df[parent_key].dropna()
    child_valid = child_df[child_key].dropna()
    
    print(f"Parent key '{parent_key}' 유효한 고유값 수: {parent_valid.nunique()}")
    print(f"Child key '{child_key}' 유효한 고유값 수: {child_valid.nunique()}")
    
    if len(parent_valid) == 0 or len(child_valid) == 0:
        print("🚨 ERROR: 유효한 키 값이 없습니다. 병합 불가능.")
        return None
    
    # 공통 키 확인 (NaN 제외)
    parent_keys = set(parent_valid)
    child_keys = set(child_valid)
    common_keys = parent_keys.intersection(child_keys)
    print(f"공통 키 개수: {len(common_keys)}")
    
    if len(common_keys) == 0:
        print("🚨 ERROR: 공통 키가 없습니다.")
        print(f"Parent 키 샘플: {list(parent_keys)[:10]}")
        print(f"Child 키 샘플: {list(child_keys)[:10]}")
        
        # 방법 1: 카테고리 인코딩을 통한 키 통일 시도
        print("방법 1: 카테고리 인코딩을 통한 키 통일 시도...")
        
        try:
            # 모든 키 값을 합쳐서 공통 카테고리 생성
            all_keys = list(parent_keys) + list(child_keys)
            unique_keys = sorted(set(all_keys))
            print(f"전체 고유 키 개수: {len(unique_keys)}")
            
            # 카테고리 생성
            key_categories = pd.Categorical(unique_keys).categories
            
            # Parent 키를 카테고리 코드로 변환
            parent_cat = pd.Categorical(parent_df[parent_key], categories=key_categories)
            parent_df_encoded = parent_df.copy()
            parent_df_encoded[parent_key + '_encoded'] = parent_cat.codes
            
            # Child 키를 카테고리 코드로 변환  
            child_cat = pd.Categorical(child_df[child_key], categories=key_categories)
            child_df_encoded = child_df.copy()
            child_df_encoded[child_key + '_encoded'] = child_cat.codes
            
            # -1 (missing category) 제거
            parent_df_encoded = parent_df_encoded[parent_df_encoded[parent_key + '_encoded'] != -1]
            child_df_encoded = child_df_encoded[child_df_encoded[child_key + '_encoded'] != -1]
            
            print(f"인코딩 후 Parent 크기: {len(parent_df_encoded)} 행")
            print(f"인코딩 후 Child 크기: {len(child_df_encoded)} 행")
            
            if len(parent_df_encoded) > 0 and len(child_df_encoded) > 0:
                # 인코딩된 키로 공통 키 확인
                parent_encoded_keys = set(parent_df_encoded[parent_key + '_encoded'])
                child_encoded_keys = set(child_df_encoded[child_key + '_encoded'])
                common_encoded_keys = parent_encoded_keys.intersection(child_encoded_keys)
                
                print(f"인코딩 후 공통 키 개수: {len(common_encoded_keys)}")
                
                if len(common_encoded_keys) > 0:
                    # 인코딩된 키로 병합
                    merged_df = pd.merge(
                        child_df_encoded, 
                        parent_df_encoded, 
                        left_on=child_key + '_encoded', 
                        right_on=parent_key + '_encoded', 
                        how='inner'
                    )
                    
                    print(f"카테고리 인코딩 후 병합 성공: {len(merged_df)}행 생성")
                    
                    if len(merged_df) > 0:
                        # 원본 키 이름으로 복구 (Parent 키 사용)
                        merged_df[parent_key] = merged_df[parent_key + '_x']  # Parent에서 온 원본 키
                        
                        # 불필요한 컬럼 제거
                        cols_to_drop = [col for col in merged_df.columns if col.endswith('_encoded') or col.endswith('_x') or col.endswith('_y')]
                        merged_df = merged_df.drop(columns=[col for col in cols_to_drop if col in merged_df.columns])
                        
                        # 정렬
                        if datetime_col and datetime_col in merged_df.columns:
                            merged_df = merged_df.sort_values(by=[parent_key, datetime_col]).reset_index(drop=True)
                        else:
                            merged_df = merged_df.sort_values(by=[parent_key]).reset_index(drop=True)
                        
                        return merged_df, parent_key, datetime_col
                    
        except Exception as e:
            print(f"카테고리 인코딩 실패: {e}")
        
        # 방법 2: 순서 기반 매핑 시도
        print("방법 2: 순서 기반 키 매핑 시도...")
        
        # Parent와 Child의 키를 정렬하여 순서 기반 매핑 시도
        parent_sorted_keys = sorted(list(parent_keys))
        child_sorted_keys = sorted(list(child_keys))
        
        min_keys = min(len(parent_sorted_keys), len(child_sorted_keys))
        
        if min_keys > 0:
            print(f"순서 기반 키 매핑 시도: {min_keys}개 키 쌍")
            
            # 키 매핑 딕셔너리 생성
            key_mapping = {}
            for i in range(min_keys):
                key_mapping[child_sorted_keys[i]] = parent_sorted_keys[i]
            
            # Child 키를 Parent 키로 매핑
            child_df_mapped = child_df.copy()
            child_df_mapped[child_key] = child_df_mapped[child_key].map(key_mapping)
            
            # 매핑되지 않은 값 제거
            child_df_mapped = child_df_mapped.dropna(subset=[child_key])
            
            print(f"매핑 후 Child 테이블 크기: {len(child_df_mapped)} 행")
            print(f"매핑 후 Child 키 샘플: {child_df_mapped[child_key].head().tolist()}")
            
            if len(child_df_mapped) > 0:
                # 매핑된 데이터로 병합 재시도
                try:
                    merged_df = pd.merge(child_df_mapped, parent_df, left_on=child_key, right_on=parent_key, how='inner')
                    print(f"매핑 후 병합 성공: {len(merged_df)}행 생성")
                    
                    if merged_df.empty:
                        print(f"Warning: 매핑 후에도 병합 결과가 빈 데이터프레임입니다.")
                        return None
                    
                    # 정렬: datetime이 있으면 [parent_key, datetime_col], 없으면 [parent_key]만
                    if datetime_col:
                        merged_df = merged_df.sort_values(by=[parent_key, datetime_col]).reset_index(drop=True)
                    else:
                        merged_df = merged_df.sort_values(by=[parent_key]).reset_index(drop=True)
                    
                    return merged_df, parent_key, datetime_col
                    
                except Exception as e:
                    print(f"매핑 후 병합 실패: {e}")
                    return None
            else:
                print("매핑 후에도 유효한 데이터가 없습니다.")
                return None
        else:
            print("키 매핑 불가능: 유효한 키가 없습니다.")
            return None
    
    # 중복 키 확인 (NaN 제외)
    parent_duplicated = parent_valid.duplicated().sum()
    child_duplicated = child_valid.duplicated().sum()
    print(f"Parent key 중복 개수: {parent_duplicated}")
    print(f"Child key 중복 개수: {child_duplicated}")
    
    if parent_duplicated > 0 or child_duplicated > 0:
        print("⚠️ WARNING: 키에 중복이 있어 Cartesian Product가 발생할 수 있음")
        
        # 예상 병합 결과 크기 계산
        if len(common_keys) > 0:
            # 각 공통 키별 예상 매칭 수 계산 (처음 10개만 샘플)
            sample_keys = list(common_keys)[:10]
            total_expected = 0
            for key in sample_keys:
                parent_count = (parent_valid == key).sum()
                child_count = (child_valid == key).sum()
                total_expected += parent_count * child_count
            
            avg_per_key = total_expected / len(sample_keys) if len(sample_keys) > 0 else 0
            estimated_total = avg_per_key * len(common_keys)
            print(f"예상 병합 결과 크기 (추정): {estimated_total:.0f} 행")
            
            if estimated_total > 1000000:  # 100만 행 이상이면 경고
                print("🚨 ERROR: 예상 병합 결과가 너무 큼. 데이터 관계를 재검토 필요")
                print("병합을 중단합니다.")
                return None

    datetime_col, datetime_format = metadata.get_datetime_col_info(child_table_name)
        
    if datetime_col is None:
        print(f"!! {child_table_name}에 datetime_col이 없음. PKey 기반 정렬을 사용합니다.")
    else:
        print(f"datetime_col: {datetime_col}, format: {datetime_format}")
    
    # datetime 컬럼 처리 (있는 경우만)
    if datetime_col and datetime_col in child_df.columns:
        try:
            child_df[datetime_col] = pd.to_datetime(child_df[datetime_col], format=datetime_format)
        except Exception as e:
            print(f"Warning: datetime 변환 실패 ({e}), datetime_col을 None으로 설정")
            datetime_col = None

    try:
        merged_df = pd.merge(child_df, parent_df, left_on=child_key, right_on=parent_key, how='inner')
        print(f"병합 성공: {len(merged_df)}행 생성")
        
        if merged_df.empty:
            print(f"Warning: 병합 결과가 빈 데이터프레임입니다.")
            return None
            
    except KeyError as e:
        print(f"Error: 병합 중 KeyError 발생: {e}")
        print(f"확인: parent_key='{parent_key}', child_key='{child_key}'")
        return None
    
    # 정렬: datetime이 있으면 [parent_key, datetime_col], 없으면 [parent_key]만
    if datetime_col:
        merged_df = merged_df.sort_values(by=[parent_key, datetime_col]).reset_index(drop=True)
    else:
        merged_df = merged_df.sort_values(by=[parent_key]).reset_index(drop=True)
    
    return merged_df, parent_key, datetime_col

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
    
    try:
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
    except Exception as e:
        print(f"no datetime column detected: {e}")
    
    if len(all_diffs) == 0:
        print(f"Warning: {numeric_col}에 대한 lag-1 차분이 계산되지 않음")
    
    return np.array(all_diffs)