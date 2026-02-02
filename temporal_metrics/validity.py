import pandas as pd
from typing import Dict, Optional

def evaluate_temporal_validity(
    real_df: pd.DataFrame, 
    synth_df: pd.DataFrame,
    time_column: Optional[str] = None,
) -> Dict:
    """
    합성 데이터가 원본 데이터에 존재하는 시점(Timestamp)만을 가지고 있는지 확인
    """
    
    results = {
        "invalid_date_count": 0,
        "invalid_rows_ratio": 0.0,
        "support_jaccard": 0.0,
        "sample_invalid_dates": []
    }
    
    # 날짜만 추출 (시간 정보 제거)
    real_dates = pd.to_datetime(real_df[time_column]).dt.normalize()
    synth_dates = pd.to_datetime(synth_df[time_column]).dt.normalize()
    
    # 1. 유효한 날짜 집합 (Set)
    valid_dates = set(real_dates.unique())
    synth_unique_dates = set(synth_dates.unique())
    
    # 2. 잘못된 날짜(Invalid Dates) 계산
    invalid_dates = synth_unique_dates - valid_dates
    results['invalid_date_count'] = len(invalid_dates)
    
    # 3. 전체 합성 데이터 레코드 중 잘못된 날짜를 가진 행의 비율 (Severity)
    invalid_rows_count = synth_dates.isin(invalid_dates).sum()
    total_rows = len(synth_df)
    invalid_ratio = invalid_rows_count / total_rows if total_rows > 0 else 0.0
    results['invalid_rows_ratio'] = invalid_ratio
    
    # 4. Support Jaccard Index (존재하는 시점의 집합 유사도)
    intersection = len(valid_dates.intersection(synth_unique_dates))
    union = len(valid_dates.union(synth_unique_dates))
    support_jaccard = intersection / union if union > 0 else 0.0
    results['support_jaccard'] = 1.0 - support_jaccard
    results['sample_invalid_dates'] = list(invalid_dates)[:5]

    return results