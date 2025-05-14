import os
import numpy as np
import pandas as pd
import re
from collections import defaultdict

# 结果目录
results_dir = './results_noise'

# 指标名称
metrics_names = ['mae', 'mse', 'rmse', 'mape', 'mspe']

# 输出目录
output_dir = './result_summary_noise'
os.makedirs(output_dir, exist_ok=True)

# 存储所有结果的字典
all_results = defaultdict(lambda: defaultdict(list))
# 新增: 存储按噪声级别分组的结果
noise_level_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# 新增: 存储按信号类型分组的结果
signal_type_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# 新增: 存储按预测长度分组的结果
pred_length_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# 新增: 存储按数据集长度分组的结果
data_length_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
# 遍历results目录下的所有文件夹
print("正在收集模型结果...")
for folder in os.listdir(results_dir):
    # 更新正则表达式以捕获预测长度信息
    match = re.search(r'long_term_forecast_(dataset\d+)_([^_]+)_([^_]+)_length(\d+)_96_(\d+)_([^_]+)', folder)
    if match:
        dataset_id = match.group(1)          # 例如 dataset1
        signal_type = match.group(2)         # 例如 Triple_Sin
        noise_level = match.group(3)         # 例如 SNR20dB 或 raw
        data_length = match.group(4)         # 例如 5000
        pred_length = match.group(5)         # 预测长度，如 10, 24, 48 等
        model_name = match.group(6)          # 模型名称
        
        # 构建数据集类型标识符
        dataset_type = f"{signal_type}_{noise_level}_length{data_length}_96_{pred_length}"
        
        # 加载指标文件
        metrics_file = os.path.join(results_dir, folder, 'metrics.npy')
        if os.path.exists(metrics_file):
            metrics = np.load(metrics_file)
            
            # 将结果添加到字典中
            for i, metric_name in enumerate(metrics_names):
                # 原始结果集
                all_results[dataset_type][f"{model_name}_{metric_name}"].append(metrics[i])
                
                # 按噪声级别分组的结果
                noise_level_results[noise_level][signal_type][f"{model_name}_{metric_name}"].append(metrics[i])
                
                # 按信号类型分组的结果
                signal_type_results[signal_type][noise_level][f"{model_name}_{metric_name}"].append(metrics[i])
                
                # 按预测长度分组的结果
                pred_length_results[pred_length][signal_type][f"{model_name}_{metric_name}"].append(metrics[i])
                
                # 按数据集长度分组的结果
                data_length_results[data_length][signal_type][f"{model_name}_{metric_name}"].append(metrics[i])
        print(f"处理: {folder}")

# 创建结果DataFrame
results_df = pd.DataFrame()

# 对每种数据集类型，计算平均指标
for dataset_type, model_metrics in all_results.items():
    row = {'dataset': dataset_type}
    
    # 计算每个模型每个指标的平均值
    for model_metric, values in model_metrics.items():
        row[model_metric] = np.mean(values)
    
    # 添加到DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

# 重新组织DataFrame以便更好地比较
# 创建一个更结构化的DataFrame
structured_results = []

for _, row in results_df.iterrows():
    dataset = row['dataset']
    
    # 提取所有模型名称
    model_names = set()
    for col in row.index:
        if col != 'dataset':
            model_name = col.split('_')[0]
            model_names.add(model_name)
    
    # 为每个模型创建一行
    for model in sorted(model_names):
        model_row = {'dataset': dataset, 'model': model}
        
        # 添加所有指标
        for metric in metrics_names:
            col_name = f"{model}_{metric}"
            if col_name in row:
                model_row[metric] = row[col_name]
        
        structured_results.append(model_row)

# 创建最终的DataFrame
final_df = pd.DataFrame(structured_results)

# 按数据集和模型排序
final_df = final_df.sort_values(['dataset', 'model'])

# 保存为CSV
final_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
print(f"已保存完整比较结果到: {os.path.join(output_dir, 'model_comparison.csv')}")

# 创建透视表格式的CSV（更易于比较）
for metric in metrics_names:
    metric_df = final_df[['dataset', 'model', metric]].pivot(index='model', columns='dataset', values=metric)
    metric_df.to_csv(os.path.join(output_dir, f'{metric}_comparison.csv'))
    
    # 打印最佳模型
    print(f"\n最佳模型 ({metric}):")
    for col in metric_df.columns:
        if metric in ['mse', 'rmse', 'mae', 'mape', 'mspe']:  # 这些指标越小越好
            best_model = metric_df[col].idxmin()
            best_value = metric_df[col].min()
        else:  # 其他指标可能越大越好
            best_model = metric_df[col].idxmax()
            best_value = metric_df[col].max()
        
        print(f"数据集 {col}: {best_model} ({best_value:.4f})")

# 创建一个汇总表，显示每个数据集上每个指标的最佳模型
summary_rows = []
for metric in metrics_names:
    metric_df = final_df[['dataset', 'model', metric]].pivot(index='model', columns='dataset', values=metric)
    
    for col in metric_df.columns:
        if metric in ['mse', 'rmse', 'mae', 'mape', 'mspe']:
            best_model = metric_df[col].idxmin()
            best_value = metric_df[col].min()
        else:
            best_model = metric_df[col].idxmax()
            best_value = metric_df[col].max()
            
        summary_rows.append({
            'dataset': col,
            'metric': metric,
            'best_model': best_model,
            'value': best_value
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(output_dir, 'best_models_summary.csv'), index=False)
print(f"\n已保存最佳模型汇总到: {os.path.join(output_dir, 'best_models_summary.csv')}")

# 新增: 按噪声级别生成比较
print("\n按噪声级别生成比较报告...")

# 对每个噪声级别创建一个DataFrame
for noise_level, signal_data in noise_level_results.items():
    noise_df = pd.DataFrame()
    
    # 处理每种信号类型
    for signal_type, model_metrics in signal_data.items():
        row = {'signal_type': signal_type}
        
        # 计算每个模型每个指标的平均值
        for model_metric, values in model_metrics.items():
            row[model_metric] = np.mean(values)
        
        # 添加到DataFrame
        noise_df = pd.concat([noise_df, pd.DataFrame([row])], ignore_index=True)
    
    # 创建结构化的结果
    noise_structured = []
    for _, row in noise_df.iterrows():
        signal_type = row['signal_type']
        
        # 提取所有模型名称
        model_names = set()
        for col in row.index:
            if col != 'signal_type':
                model_name = col.split('_')[0]
                model_names.add(model_name)
        
        # 为每个模型创建一行
        for model in sorted(model_names):
            model_row = {'signal_type': signal_type, 'model': model}
            
            # 添加所有指标
            for metric in metrics_names:
                col_name = f"{model}_{metric}"
                if col_name in row:
                    model_row[metric] = row[col_name]
            
            noise_structured.append(model_row)
    
    # 创建最终的噪声级别DataFrame
    noise_final_df = pd.DataFrame(noise_structured)
    noise_final_df = noise_final_df.sort_values(['signal_type', 'model'])
    
    # 保存为CSV
    noise_file = os.path.join(output_dir, f'noise_level_{noise_level}_comparison.csv')
    noise_final_df.to_csv(noise_file, index=False)
    print(f"已保存噪声级别 {noise_level} 比较结果到: {noise_file}")
    
    # 为每个指标创建透视表
    for metric in metrics_names:
        metric_pivot = noise_final_df[['signal_type', 'model', metric]].pivot(
            index='model', columns='signal_type', values=metric)
        metric_file = os.path.join(output_dir, f'noise_level_{noise_level}_{metric}_comparison.csv')
        metric_pivot.to_csv(metric_file)
        print(f"已保存噪声级别 {noise_level} 的 {metric} 指标比较到: {metric_file}")

# 新增: 按信号类型生成比较
print("\n按信号类型生成比较报告...")

# 对每个信号类型创建一个DataFrame，比较不同噪声级别的性能
for signal_type, noise_data in signal_type_results.items():
    signal_df = pd.DataFrame()
    
    # 处理每种噪声级别
    for noise_level, model_metrics in noise_data.items():
        row = {'noise_level': noise_level}
        
        # 计算每个模型每个指标的平均值
        for model_metric, values in model_metrics.items():
            row[model_metric] = np.mean(values)
        
        # 添加到DataFrame
        signal_df = pd.concat([signal_df, pd.DataFrame([row])], ignore_index=True)
    
    # 创建结构化的结果
    signal_structured = []
    for _, row in signal_df.iterrows():
        noise_level = row['noise_level']
        
        # 提取所有模型名称
        model_names = set()
        for col in row.index:
            if col != 'noise_level':
                model_name = col.split('_')[0]
                model_names.add(model_name)
        
        # 为每个模型创建一行
        for model in sorted(model_names):
            model_row = {'noise_level': noise_level, 'model': model}
            
            # 添加所有指标
            for metric in metrics_names:
                col_name = f"{model}_{metric}"
                if col_name in row:
                    model_row[metric] = row[col_name]
            
            signal_structured.append(model_row)
    
    # 创建最终的信号类型DataFrame
    signal_final_df = pd.DataFrame(signal_structured)
    signal_final_df = signal_final_df.sort_values(['noise_level', 'model'])
    
    # 保存为CSV
    signal_file = os.path.join(output_dir, f'signal_type_{signal_type}_comparison.csv')
    signal_final_df.to_csv(signal_file, index=False)
    print(f"已保存信号类型 {signal_type} 比较结果到: {signal_file}")
    
    # 为每个指标创建透视表
    for metric in metrics_names:
        metric_pivot = signal_final_df[['noise_level', 'model', metric]].pivot(
            index='model', columns='noise_level', values=metric)
        metric_file = os.path.join(output_dir, f'signal_type_{signal_type}_{metric}_comparison.csv')
        metric_pivot.to_csv(metric_file)
        print(f"已保存信号类型 {signal_type} 的 {metric} 指标比较到: {metric_file}")

# 创建噪声级别性能敏感度分析
print("\n生成噪声级别敏感度分析...")

# 合并所有模型在所有信号类型上的表现，按噪声级别分组
noise_sensitivity = defaultdict(lambda: defaultdict(list))

for signal_type, noise_data in signal_type_results.items():
    for noise_level, model_metrics in noise_data.items():
        for model_metric, values in model_metrics.items():
            model_name, metric = model_metric.split('_', 1)
            avg_value = np.mean(values)
            noise_sensitivity[noise_level][f"{model_name}_{metric}"].append(avg_value)

# 计算每个噪声级别下模型的平均表现
noise_sensitivity_df = pd.DataFrame()
for noise_level, metrics in noise_sensitivity.items():
    row = {'noise_level': noise_level}
    for model_metric, values in metrics.items():
        row[model_metric] = np.mean(values)
    noise_sensitivity_df = pd.concat([noise_sensitivity_df, pd.DataFrame([row])], ignore_index=True)

# 保存噪声敏感度分析结果
noise_sensitivity_file = os.path.join(output_dir, 'noise_sensitivity_analysis.csv')
noise_sensitivity_df.to_csv(noise_sensitivity_file, index=False)
print(f"已保存噪声敏感度分析到: {noise_sensitivity_file}")

# 创建预测长度性能分析
print("\n生成预测长度性能分析...")

# 对每个预测长度创建一个DataFrame
for pred_length, signal_data in pred_length_results.items():
    pred_df = pd.DataFrame()
    
    # 处理每种信号类型
    for signal_type, model_metrics in signal_data.items():
        row = {'signal_type': signal_type}
        
        # 计算每个模型每个指标的平均值
        for model_metric, values in model_metrics.items():
            row[model_metric] = np.mean(values)
        
        # 添加到DataFrame
        pred_df = pd.concat([pred_df, pd.DataFrame([row])], ignore_index=True)
    
    # 创建结构化的结果
    pred_structured = []
    for _, row in pred_df.iterrows():
        signal_type = row['signal_type']
        
        # 提取所有模型名称
        model_names = set()
        for col in row.index:
            if col != 'signal_type':
                model_name = col.split('_')[0]
                model_names.add(model_name)
        
        # 为每个模型创建一行
        for model in sorted(model_names):
            model_row = {'signal_type': signal_type, 'model': model}
            
            # 添加所有指标
            for metric in metrics_names:
                col_name = f"{model}_{metric}"
                if col_name in row:
                    model_row[metric] = row[col_name]
            
            pred_structured.append(model_row)
    
    # 创建最终的预测长度DataFrame
    pred_final_df = pd.DataFrame(pred_structured)
    pred_final_df = pred_final_df.sort_values(['signal_type', 'model'])
    
    # 保存为CSV
    pred_file = os.path.join(output_dir, f'pred_length_{pred_length}_comparison.csv')
    pred_final_df.to_csv(pred_file, index=False)
    print(f"已保存预测长度 {pred_length} 比较结果到: {pred_file}")
    
    # 为每个指标创建透视表
    for metric in metrics_names:
        metric_pivot = pred_final_df[['signal_type', 'model', metric]].pivot(
            index='model', columns='signal_type', values=metric)
        metric_file = os.path.join(output_dir, f'pred_length_{pred_length}_{metric}_comparison.csv')
        metric_pivot.to_csv(metric_file)
        print(f"已保存预测长度 {pred_length} 的 {metric} 指标比较到: {metric_file}")

# 创建预测长度敏感度分析 - 修复版
print("\n生成预测长度敏感度分析...")

# 重置预测长度敏感度分析数据结构
pred_sensitivity = defaultdict(lambda: defaultdict(list))

# 直接从结果文件夹中读取数据，按预测长度分组
for folder in os.listdir(results_dir):
    match = re.search(r'long_term_forecast_(dataset\d+)_([^_]+)_([^_]+)_length(\d+)_96_(\d+)_([^_]+)', folder)
    if match:
        dataset_id = match.group(1)
        signal_type = match.group(2)
        noise_level = match.group(3)
        data_length = match.group(4)
        pred_length = match.group(5)  # 预测长度
        model_name = match.group(6)
        
        # 加载指标文件
        metrics_file = os.path.join(results_dir, folder, 'metrics.npy')
        if os.path.exists(metrics_file):
            metrics = np.load(metrics_file)
            
            # 将每个指标值添加到对应预测长度的数据集中
            for i, metric_name in enumerate(metrics_names):
                pred_sensitivity[pred_length][f"{model_name}_{metric_name}"].append(metrics[i])

# 计算每个预测长度下模型的平均表现
pred_sensitivity_df = pd.DataFrame()
for pred_len, metrics in pred_sensitivity.items():
    row = {'prediction_length': pred_len}
    for model_metric, values in metrics.items():
        row[model_metric] = np.mean(values)
    pred_sensitivity_df = pd.concat([pred_sensitivity_df, pd.DataFrame([row])], ignore_index=True)

# 保存预测长度敏感度分析结果
pred_sensitivity_file = os.path.join(output_dir, 'prediction_length_sensitivity_analysis.csv')
pred_sensitivity_df.to_csv(pred_sensitivity_file, index=False)
print(f"已保存预测长度敏感度分析到: {pred_sensitivity_file}")



# 在pred_sensitivity分析后添加

# 创建数据集长度性能分析
print("\n生成数据集长度性能分析...")

# 对每个数据集长度创建一个DataFrame
for data_length, signal_data in data_length_results.items():
    data_len_df = pd.DataFrame()
    
    # 处理每种信号类型
    for signal_type, model_metrics in signal_data.items():
        row = {'signal_type': signal_type}
        
        # 计算每个模型每个指标的平均值
        for model_metric, values in model_metrics.items():
            row[model_metric] = np.mean(values)
        
        # 添加到DataFrame
        data_len_df = pd.concat([data_len_df, pd.DataFrame([row])], ignore_index=True)
    
    # 创建结构化的结果
    data_len_structured = []
    for _, row in data_len_df.iterrows():
        signal_type = row['signal_type']
        
        # 提取所有模型名称
        model_names = set()
        for col in row.index:
            if col != 'signal_type':
                model_name = col.split('_')[0]
                model_names.add(model_name)
        
        # 为每个模型创建一行
        for model in sorted(model_names):
            model_row = {'signal_type': signal_type, 'model': model}
            
            # 添加所有指标
            for metric in metrics_names:
                col_name = f"{model}_{metric}"
                if col_name in row:
                    model_row[metric] = row[col_name]
            
            data_len_structured.append(model_row)
    
    # 创建最终的数据集长度DataFrame
    data_len_final_df = pd.DataFrame(data_len_structured)
    data_len_final_df = data_len_final_df.sort_values(['signal_type', 'model'])
    
    # 保存为CSV
    data_len_file = os.path.join(output_dir, f'data_length_{data_length}_comparison.csv')
    data_len_final_df.to_csv(data_len_file, index=False)
    print(f"已保存数据集长度 {data_length} 比较结果到: {data_len_file}")
    
    # 为每个指标创建透视表
    for metric in metrics_names:
        metric_pivot = data_len_final_df[['signal_type', 'model', metric]].pivot(
            index='model', columns='signal_type', values=metric)
        metric_file = os.path.join(output_dir, f'data_length_{data_length}_{metric}_comparison.csv')
        metric_pivot.to_csv(metric_file)
        print(f"已保存数据集长度 {data_length} 的 {metric} 指标比较到: {metric_file}")

# 创建数据集长度敏感度分析
print("\n生成数据集长度敏感度分析...")

# 合并所有模型在所有信号类型上的表现，按数据集长度分组
data_len_sensitivity = defaultdict(lambda: defaultdict(list))

for signal_type, noise_data in signal_type_results.items():
    for noise_level, model_metrics in noise_data.items():
        for model_metric, values in model_metrics.items():
            model_name, metric = model_metric.split('_', 1)
            avg_value = np.mean(values)
            data_len_info = [folder for folder in os.listdir(results_dir) 
                        if signal_type in folder and noise_level in folder and model_name in folder]
            for folder in data_len_info:
                match = re.search(r'.*_length(\d+)_.*', folder)
                if match:
                    data_len = match.group(1)
                    data_len_sensitivity[data_len][f"{model_name}_{metric}"].append(avg_value)

# 计算每个数据集长度下模型的平均表现
data_len_sensitivity_df = pd.DataFrame()
for data_len, metrics in data_len_sensitivity.items():
    row = {'data_length': data_len}
    for model_metric, values in metrics.items():
        row[model_metric] = np.mean(values)
    data_len_sensitivity_df = pd.concat([data_len_sensitivity_df, pd.DataFrame([row])], ignore_index=True)

# 保存数据集长度敏感度分析结果
data_len_sensitivity_file = os.path.join(output_dir, 'data_length_sensitivity_analysis.csv')
data_len_sensitivity_df.to_csv(data_len_sensitivity_file, index=False)
print(f"已保存数据集长度敏感度分析到: {data_len_sensitivity_file}")


print("\n生成综合性能汇总表...")

# Dictionary to store aggregated results by signal type and model
comprehensive_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Loop through all result folders to collect data
for folder in os.listdir(results_dir):
    match = re.search(r'long_term_forecast_(dataset\d+)_([^_]+)_([^_]+)_length(\d+)_96_(\d+)_([^_]+)', folder)
    if match:
        dataset_id = match.group(1)          # e.g., dataset1
        signal_type = match.group(2)         # e.g., Triple_Sin
        noise_level = match.group(3)         # e.g., SNR20dB or raw
        data_length = match.group(4)         # e.g., 5000
        pred_length = match.group(5)         # prediction length
        model_name = match.group(6)          # model name
        
        # Load metrics file
        metrics_file = os.path.join(results_dir, folder, 'metrics.npy')
        if os.path.exists(metrics_file):
            metrics = np.load(metrics_file)
            
            # Add metrics to the comprehensive results dictionary
            for i, metric_name in enumerate(metrics_names):
                comprehensive_results[signal_type][model_name][metric_name].append(metrics[i])

# Create a list to store rows for our final DataFrame
comprehensive_summary_rows = []

# Process the collected results
for signal_type, model_data in comprehensive_results.items():
    for model_name, metrics_data in model_data.items():
        # Create a row with signal_type and model_name
        row = {'signal_type': signal_type, 'model': model_name}
        
        # Calculate average for each metric
        for metric_name, values in metrics_data.items():
            if values:  # Check if we have values to average
                row[metric_name] = np.mean(values)
            else:
                row[metric_name] = np.nan  # Handle empty lists
        
        comprehensive_summary_rows.append(row)

# Create and save the comprehensive summary DataFrame
comprehensive_df = pd.DataFrame(comprehensive_summary_rows)
comprehensive_df = comprehensive_df.sort_values(['signal_type', 'model'])
comprehensive_file = os.path.join(output_dir, 'comprehensive_model_summary.csv')
comprehensive_df.to_csv(comprehensive_file, index=False)
print(f"已保存综合性能汇总表到: {comprehensive_file}")

# Also create a pivot table for easier comparison
for metric in metrics_names:
    if metric in comprehensive_df.columns:  # Check if metric exists in df
        metric_pivot = comprehensive_df[['signal_type', 'model', metric]].pivot(
            index='signal_type', columns='model', values=metric)
        pivot_file = os.path.join(output_dir, f'comprehensive_{metric}_comparison.csv')
        metric_pivot.to_csv(pivot_file)
        print(f"已保存{metric}指标的信号类型-模型对比表到: {pivot_file}")
