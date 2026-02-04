import temporal_benchmark
import utils
import pandas as pd
import numpy as np
import os

def aggregate_metrics(results_list):
    """
    여러 관계(테이블)에서 나온 metric 딕셔너리들의 리스트를 받아 평균을 계산합니다.
    """
    if not results_list:
        return {}
    
    # 결과가 하나뿐이면(Rossmann 등) 그대로 반환
    if len(results_list) == 1:
        return results_list[0]
    
    aggregated = {}
    # 첫 번째 결과의 키를 기준으로 순회
    all_keys = results_list[0].keys()
    
    for key in all_keys:
        # 해당 키에 대한 모든 결과값 수집
        values = [res.get(key) for res in results_list]
        
        # 1. 값이 모두 딕셔너리인 경우 -> 재귀 호출
        if all(isinstance(v, dict) for v in values):
            aggregated[key] = aggregate_metrics(values)
            
        # 2. 값이 모두 수치형(int, float)인 경우 -> 평균 계산
        elif all(isinstance(v, (int, float, np.number)) for v in values):
            aggregated[key] = float(np.mean(values))
            
        # 3. 그 외(문자열 등) -> 첫 번째 값 유지 (또는 필요시 병합 로직 추가)
        else:
            aggregated[key] = values[0]
            
    return aggregated

def run_temporal_benchmark(
    real_path: str = 'data/rossmann_subsampled_real.csv',
    synth_path: str = 'data/rossmann_subsampled_synthetic.csv',
    dataset_name: str = 'rossmann_subsampled',
    bin_strategy: str = 'weekly',
):
    all_relationships_results = []
    
    metadata = utils.load_metadata(real_path, dataset_name)

    # # 월별 binning
    # benchmark = temporal_benchmark.TemporalBenchmark(
    #     metadata=metadata,
    #     time_column='Date',
    #     bin_strategy=bin_strategy,
    # )
    
    for relationship in metadata['relationships']:
        parent_table = relationship['parent_table_name']
        child_table = relationship['child_table_name']
        
        parent_key = relationship['child_foreign_key']
        
        print(f"\n--- Relationship: {parent_table} -> {child_table} ---")

        time_col = None
        real_data_path = os.path.join(real_path, dataset_name)
        print("Merging real data...")
        real_df, _, _ = utils.load_and_preprocess_data(real_data_path, metadata, parent_table, child_table)
        print("Merging synthetic data...")
        synth_df, _, time_col = utils.load_and_preprocess_data(synth_path, metadata, parent_table, child_table)
        
        available_features = synth_df.columns

        if time_col == None : continue
        benchmark = temporal_benchmark.TemporalBenchmark(
            metadata=metadata,
            time_column=time_col,
            bin_strategy=bin_strategy,
        )  
        
        if parent_key not in real_df.columns or parent_key not in synth_df.columns:
            print(f"Warning: Parent key '{parent_key}' not found in merged dataframe")
            continue
        
        print()

        num_cols = benchmark.get_numeric_columns(real_df, available_features, parent_table) 
        num_cols.extend(benchmark.get_numeric_columns(real_df, available_features, child_table))

        cat_cols = benchmark.get_categorical_columns(real_df, available_features, parent_table) 
        cat_cols.extend(benchmark.get_categorical_columns(real_df, available_features, child_table))
        
        # """
        
        # 종합 평가 수행
        try:
            results = benchmark.comprehensive_evaluation(
                real_df=real_df,
                synth_df=synth_df,
                features=available_features,
                num_cols = num_cols,
                cat_cols = cat_cols,
                parent_key=parent_key,
            )
            
            all_relationships_results.append(results)
            print(f"✅ Completed evaluation for {child_table}")
            
        except Exception as e:
            print(f"❌ Error evaluating {child_table}: {str(e)}")


        """

        real_binned = benchmark.create_time_bins(real_df)
        syn_binned = benchmark.create_time_bins(synth_df)

        print(syn_binned['Date'].sort_values().value_counts())

        # print("찐 데이터 tiem bin nunique")
        # print(real_binned['time_bin'].value_counts())
        # print("\n짭 데이터 tiem bin nunique")
        # print(syn_binned['time_bin'].value_counts())
        
        """

    # [수정] 결과 집계 (Aggregation)
    # 관계가 1개면 그대로, 2개 이상이면 평균 계산
    final_results = aggregate_metrics(all_relationships_results)
        
    # 결과 출력
    print("\n\n" + "=" * 80)
    print(f"SUMMARY RESULTS ({len(all_relationships_results)} relationships aggregated)")
    print("=" * 80)
    # print(final_results) # 필요시 주석 해제
    
    return final_results


if __name__ == "__main__":
    
    #                 0           1             2           3        4        5
    syn_models = ['CLAVADDPM', 'RCTGAN', 'REALTABFORMER', 'RGCLD', 'SDV', 'RelDiff']
    syn_model_name = syn_models[4]
    
    # DATASET = 'rossmann_subsampled'
    DATASET = 'walmart_subsampled'
    # DATASET = 'berka'

    # 실행
    results = run_temporal_benchmark(
        real_path = "/home/yjung/syntherela/experiments/data/original/",
        synth_path  = f"/home/yjung/syntherela/experiments/data/synthetic/{DATASET}/{syn_model_name}/1/sample1/",
        dataset_name = DATASET,
        bin_strategy = 'weekly',
    )
    
    if results:
        # 결과 저장
        import json
        with open('temporal_benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("\n✅ Results saved to temporal_benchmark_results.json")        

