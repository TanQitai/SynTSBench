import os
import numpy as np
import pandas as pd
import re
from collections import defaultdict

# 结果目录
results_dir = './results_multivariate'

# 指标名称
metrics_names = ['mae', 'mse', 'rmse', 'mape', 'mspe']

# 输出目录
output_dir = './result_summary_multivariate'
os.makedirs(output_dir, exist_ok=True)

# 存储所有结果的字典 - 整体指标
overall_results = defaultdict(lambda: defaultdict(list))
# 存储每个变量指标的字典
variable_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

def calculate_metrics(pred, true):
    """计算指标 - 与utils/metrics.py中的计算方法保持一致"""
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    
    # 处理可能的除零情况
    mask = true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask]))
    else:
        mape = np.nan
    
    # 计算MSPE时添加一个小的常数避免除零
    mspe = np.mean(np.square(np.abs(true - pred) / (np.abs(true) + 1e-5)))
    
    return mae, mse, rmse, mape, mspe

print("正在收集多变量时间序列模型结果...")
for folder in os.listdir(results_dir):
    # 使用与univariate相同的正则表达式模式
    match = re.search(r'long_term_forecast_(dataset\d+)_([^_]+)_([^_]+)_length(\d+)_96_(\d+)_([^_]+)', folder)
    if match:
        dataset_id = match.group(1)          # 例如 dataset1
        signal_type = match.group(2)         # 例如 Triple_Sin
        noise_level = match.group(3)         # 例如 SNR20dB 或 raw
        data_length = match.group(4)         # 例如 5000
        pred_length = match.group(5)         # 预测长度，如 10, 24, 48 等
        model_name = match.group(6)          # 模型名称
        
        # 构建数据集类型标识符 (使用简化的标识符)
        dataset_type = f"{signal_type}_{noise_level}_{pred_length}"
        
        # 加载已计算的整体指标
        metrics_file = os.path.join(results_dir, folder, 'metrics.npy')
        if os.path.exists(metrics_file):
            overall_metrics = np.load(metrics_file)
            
            # 加载预测和真实值以计算每个变量的指标
            pred_file = os.path.join(results_dir, folder, 'pred.npy')
            true_file = os.path.join(results_dir, folder, 'true.npy')
            
            if os.path.exists(pred_file) and os.path.exists(true_file):
                preds = np.load(pred_file)  # 形状: (samples, sequence_length, n_variables)
                trues = np.load(true_file)  # 形状: (samples, sequence_length, n_variables)
                
                print(f"处理: {signal_type}_{noise_level}_{pred_length}_{model_name}")
                
                # 保存整体指标
                for i, metric_name in enumerate(metrics_names):
                    overall_results[dataset_type][f"{model_name}_{metric_name}"].append(overall_metrics[i])
                
                # 计算每个变量的指标
                n_variables = preds.shape[-1]
                for var_idx in range(n_variables):
                    var_preds = preds[:, :, var_idx]
                    var_trues = trues[:, :, var_idx]
                    
                    var_metrics = calculate_metrics(var_preds, var_trues)
                    
                    # 保存每个变量的指标
                    for i, metric_name in enumerate(metrics_names):
                        variable_results[dataset_type][f"var{var_idx+1}"][f"{model_name}_{metric_name}"].append(var_metrics[i])

# 创建整体结果DataFrame
overall_df = pd.DataFrame()
for dataset_type, metrics in overall_results.items():
    row = {'dataset': dataset_type}
    for metric_key, values in metrics.items():
        row[metric_key] = np.mean(values)
    overall_df = pd.concat([overall_df, pd.DataFrame([row])], ignore_index=True)

# 保存整体结果
overall_df.to_csv(os.path.join(output_dir, 'overall_model_comparison.csv'), index=False)
print(f"已保存整体模型比较结果到: {os.path.join(output_dir, 'overall_model_comparison.csv')}")

# 为整体结果创建结构化的DataFrame
structured_overall = []
for _, row in overall_df.iterrows():
    dataset = row['dataset']
    
    # 提取模型名称
    model_names = set()
    for col in row.index:
        if col != 'dataset':
            model_name = col.split('_')[0]
            model_names.add(model_name)
    
    # 为每个模型创建一行
    for model in sorted(model_names):
        model_row = {'dataset': dataset, 'model': model}
        
        # 添加指标
        for metric in metrics_names:
            col_name = f"{model}_{metric}"
            if col_name in row:
                model_row[metric] = row[col_name]
        
        structured_overall.append(model_row)

# 创建并保存结构化的整体结果
structured_overall_df = pd.DataFrame(structured_overall)
structured_overall_df = structured_overall_df.sort_values(['dataset', 'model'])
structured_overall_df.to_csv(os.path.join(output_dir, 'structured_overall_comparison.csv'), index=False)

# 为每个指标创建整体透视表
for metric in metrics_names:
    metric_pivot = structured_overall_df[['dataset', 'model', metric]].pivot(
        index='model', columns='dataset', values=metric)
    metric_file = os.path.join(output_dir, f'overall_{metric}_comparison.csv')
    metric_pivot.to_csv(metric_file)
    print(f"已保存整体{metric}指标比较到: {metric_file}")
    
    # 打印最佳模型
    print(f"\n最佳模型 (整体{metric}):")
    for col in metric_pivot.columns:
        if metric in ['mse', 'rmse', 'mae', 'mape', 'mspe']:  # 这些指标越小越好
            best_model = metric_pivot[col].idxmin()
            best_value = metric_pivot[col].min()
        else:  # 其他指标可能越大越好
            best_model = metric_pivot[col].idxmax()
            best_value = metric_pivot[col].max()
        
        print(f"数据集 {col}: {best_model} ({best_value:.4f})")

# 处理每个变量的结果
variable_summary = []
for dataset_type, vars_data in variable_results.items():
    for var_name, metrics in vars_data.items():
        # 创建变量结果DataFrame
        var_df = pd.DataFrame()
        
        row = {'dataset': dataset_type, 'variable': var_name}
        for metric_key, values in metrics.items():
            row[metric_key] = np.mean(values)
        
        var_df = pd.concat([var_df, pd.DataFrame([row])], ignore_index=True)
        
        # 为该变量创建结构化DataFrame
        structured_var = []
        for _, var_row in var_df.iterrows():
            # 提取模型名称
            model_names = set()
            for col in var_row.index:
                if col not in ['dataset', 'variable']:
                    model_name = col.split('_')[0]
                    model_names.add(model_name)
            
            # 为每个模型创建一行
            for model in sorted(model_names):
                model_row = {
                    'dataset': dataset_type,
                    'variable': var_name,
                    'model': model
                }
                
                # 添加指标
                for metric in metrics_names:
                    col_name = f"{model}_{metric}"
                    if col_name in var_row:
                        model_row[metric] = var_row[col_name]
                
                structured_var.append(model_row)
                variable_summary.append(model_row)
        
        # 如果存在结构化数据，则保存
        if structured_var:
            struc_var_df = pd.DataFrame(structured_var)
            var_file = os.path.join(output_dir, f'{dataset_type}_{var_name}_comparison.csv')
            struc_var_df.to_csv(var_file, index=False)
            print(f"已保存变量 {var_name} 在数据集 {dataset_type} 上的比较结果到: {var_file}")

# 创建所有变量的汇总表
all_vars_structured_df = pd.DataFrame(variable_summary)
all_vars_structured_df.to_csv(os.path.join(output_dir, 'all_variables_comparison.csv'), index=False)
print(f"已保存所有变量比较结果到: {os.path.join(output_dir, 'all_variables_comparison.csv')}")

# 找出每个数据集每个变量上表现最佳的模型
best_models_summary = []
for dataset in all_vars_structured_df['dataset'].unique():
    dataset_df = all_vars_structured_df[all_vars_structured_df['dataset'] == dataset]
    
    for variable in dataset_df['variable'].unique():
        var_df = dataset_df[dataset_df['variable'] == variable]
        
        for metric in metrics_names:
            if metric in var_df.columns:
                if metric in ['mse', 'rmse', 'mae', 'mape', 'mspe']:  # 这些指标越小越好
                    best_idx = var_df[metric].idxmin()
                    best_value = var_df.loc[best_idx, metric]
                    best_model = var_df.loc[best_idx, 'model']
                else:  # 其他指标可能越大越好
                    best_idx = var_df[metric].idxmax()
                    best_value = var_df.loc[best_idx, metric]
                    best_model = var_df.loc[best_idx, 'model']
                
                best_models_summary.append({
                    'dataset': dataset,
                    'variable': variable,
                    'metric': metric,
                    'best_model': best_model,
                    'value': best_value
                })

# 保存最佳模型汇总
best_models_df = pd.DataFrame(best_models_summary)
best_models_file = os.path.join(output_dir, 'best_models_by_variable.csv')
best_models_df.to_csv(best_models_file, index=False)
print(f"已保存变量级别的最佳模型汇总到: {best_models_file}")

# 创建综合性能汇总表
print("\n生成综合性能汇总表...")

# 计算每个信号类型和模型的平均性能
comprehensive_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for dataset_type, vars_data in variable_results.items():
    # 从dataset_type中提取信号类型
    signal_type = dataset_type.split('_')[0]
    
    for var_name, metrics in vars_data.items():
        for model_metric, values in metrics.items():
            model_name, metric = model_metric.split('_', 1)
            comprehensive_results[signal_type][model_name][metric].append(np.mean(values))

# 处理收集到的结果
comprehensive_summary_rows = []
for signal_type, model_data in comprehensive_results.items():
    for model_name, metrics_data in model_data.items():
        row = {'signal_type': signal_type, 'model': model_name}
        
        for metric_name, values in metrics_data.items():
            if values:  # 检查是否有值可以计算平均值
                row[metric_name] = np.mean(values)
            else:
                row[metric_name] = np.nan  # 处理空列表
        
        comprehensive_summary_rows.append(row)

# 创建并保存综合汇总DataFrame
comprehensive_df = pd.DataFrame(comprehensive_summary_rows)
comprehensive_df = comprehensive_df.sort_values(['signal_type', 'model'])
comprehensive_file = os.path.join(output_dir, 'comprehensive_model_summary.csv')
comprehensive_df.to_csv(comprehensive_file, index=False)
print(f"已保存综合性能汇总表到: {comprehensive_file}")

# 为每个指标创建透视表以方便比较
for metric in metrics_names:
    if metric in comprehensive_df.columns:  # 检查指标是否存在于df中
        metric_pivot = comprehensive_df[['signal_type', 'model', metric]].pivot(
            index='signal_type', columns='model', values=metric)
        pivot_file = os.path.join(output_dir, f'comprehensive_{metric}_comparison.csv')
        metric_pivot.to_csv(pivot_file)
        print(f"已保存{metric}指标的信号类型-模型对比表到: {pivot_file}")

# After creating pivot tables for each metric, add this code to save the best models for all metrics

# 创建每个指标的最佳模型汇总
print("\n生成所有指标的最佳模型汇总...")

# 为每个指标生成最佳模型汇总
for metric in metrics_names:
    best_models = []
    
    # 创建该指标的透视表
    metric_pivot = structured_overall_df[['dataset', 'model', metric]].pivot(
        index='model', columns='dataset', values=metric)
    
    for dataset in metric_pivot.columns:
        if metric in ['mse', 'rmse', 'mae', 'mape', 'mspe']:  # 这些指标越小越好
            best_model = metric_pivot[dataset].idxmin()
            best_value = metric_pivot[dataset].min()
        else:  # 其他指标可能越大越好
            best_model = metric_pivot[dataset].idxmax()
            best_value = metric_pivot[dataset].max()
        
        best_models.append({
            'dataset': dataset,
            'best_model': best_model,
            metric: best_value
        })
    
    # 创建并保存该指标的最佳模型汇总
    best_models_df = pd.DataFrame(best_models)
    best_models_file = os.path.join(output_dir, f'best_models_{metric}_overall.csv')
    best_models_df.to_csv(best_models_file, index=False)
    print(f"已保存{metric}指标的最佳模型汇总到: {best_models_file}")
    
    # 可视化展示每个数据集的最佳模型
    print(f"\n总体{metric}最佳模型汇总:")
    for _, row in best_models_df.iterrows():
        print(f"数据集 {row['dataset']}: {row['best_model']} ({metric}: {row[metric]:.4f})")

# 创建一个综合的最佳模型汇总表格
print("\n生成综合最佳模型汇总表格...")
comprehensive_best = defaultdict(dict)

# 收集每个数据集在每个指标上的最佳模型
for dataset in structured_overall_df['dataset'].unique():
    comprehensive_best[dataset] = {'dataset': dataset}
    dataset_df = structured_overall_df[structured_overall_df['dataset'] == dataset]
    
    for metric in metrics_names:
        if metric in dataset_df.columns:
            if metric in ['mse', 'rmse', 'mae', 'mape', 'mspe']:  # 这些指标越小越好
                best_idx = dataset_df[metric].idxmin()
                best_value = dataset_df.loc[best_idx, metric]
                best_model = dataset_df.loc[best_idx, 'model']
            else:  # 其他指标可能越大越好
                best_idx = dataset_df[metric].idxmax()
                best_value = dataset_df.loc[best_idx, metric]
                best_model = dataset_df.loc[best_idx, 'model']
            
            comprehensive_best[dataset][f'best_model_{metric}'] = best_model
            comprehensive_best[dataset][f'best_{metric}'] = best_value

# 创建并保存综合最佳模型汇总
comprehensive_best_df = pd.DataFrame(list(comprehensive_best.values()))
comprehensive_best_file = os.path.join(output_dir, 'comprehensive_best_models.csv')
comprehensive_best_df.to_csv(comprehensive_best_file, index=False)
print(f"已保存综合最佳模型汇总到: {comprehensive_best_file}")

print("\n结果收集完成！")