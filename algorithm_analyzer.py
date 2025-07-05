#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""算法性能分析脚本
根据 avg_latency + load + avg_compute_time 的值评判算法效果
"""

import re
import json
from collections import defaultdict

def parse_test_data(file_path):
    """
    解析测试数据文件
    """
    results = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式匹配每个测试结果块
    pattern = r'(\w+)_([^_]+)_(\d+)\{([^}]+)\}'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        algorithm = match[0]
        topology = match[1]
        controller_num = int(match[2])
        data_block = match[3]
        
        # 解析数据块中的字段
        result = {
            'algorithm': algorithm,
            'topology': topology,
            'controller_num': controller_num
        }
        
        # 提取各个字段的值
        fields = ['avg_latency', 'worst_latency', 'load', 'avg_compute_time']
        for field in fields:
            # 匹配字段值，支持科学计数法
            field_pattern = rf'"{field}"\s*:\s*([\d\.e\-\+]+)'
            field_match = re.search(field_pattern, data_block)
            if field_match:
                result[field] = float(field_match.group(1))
            else:
                # 如果没有引号，尝试不带引号的格式
                field_pattern = rf'{field}"\s*:\s*([\d\.e\-\+]+)'
                field_match = re.search(field_pattern, data_block)
                if field_match:
                    result[field] = float(field_match.group(1))
                else:
                    result[field] = 0.0
        
        # 计算评判标准：avg_latency + load + avg_compute_time
        result['score'] = result['avg_latency'] + result['load'] + result['avg_compute_time']
        
        results.append(result)
    
    return results

def analyze_algorithms(results):
    """
    分析算法性能
    """
    # 按拓扑和控制器数量分组
    grouped = defaultdict(list)
    
    for result in results:
        key = (result['topology'], result['controller_num'])
        grouped[key].append(result)
    
    # 找出每个配置下的最佳算法
    best_algorithms = {}
    all_scores = []
    
    print("=" * 80)
    print("各配置下的算法性能对比 (评判标准: avg_latency + load + avg_compute_time, 越小越好)")
    print("=" * 80)
    
    for (topology, controller_num), group in sorted(grouped.items()):
        print(f"\n拓扑: {topology}, 控制器数量: {controller_num}")
        print("-" * 60)
        
        # 按分数排序
        group_sorted = sorted(group, key=lambda x: x['score'])
        
        for i, result in enumerate(group_sorted):
            status = "★ 最佳" if i == 0 else f"  第{i+1}名"
            print(f"{status} {result['algorithm']:10} | "
                  f"avg_latency: {result['avg_latency']:8.4f} | "
                  f"load: {result['load']:8.4f} | "
                  f"avg_compute_time: {result['avg_compute_time']:8.4f} | "
                  f"总分: {result['score']:8.4f}")
        
        # 记录最佳算法
        best = group_sorted[0]
        best_algorithms[(topology, controller_num)] = best
        all_scores.extend([r['score'] for r in group])
    
    return best_algorithms, all_scores

def summarize_results(best_algorithms):
    """
    汇总分析结果
    """
    # 统计各算法获胜次数
    algorithm_wins = defaultdict(int)
    topology_stats = defaultdict(list)
    controller_stats = defaultdict(list)
    
    for (topology, controller_num), best in best_algorithms.items():
        algorithm_wins[best['algorithm']] += 1
        topology_stats[topology].append(best)
        controller_stats[controller_num].append(best)
    
    print("\n" + "=" * 80)
    print("算法性能汇总分析")
    print("=" * 80)
    
    # 算法获胜统计
    print("\n各算法获胜次数统计:")
    print("-" * 30)
    for algorithm, wins in sorted(algorithm_wins.items(), key=lambda x: x[1], reverse=True):
        total_configs = len(best_algorithms)
        win_rate = wins / total_configs * 100
        print(f"{algorithm:10}: {wins:2d}次 ({win_rate:5.1f}%)")
    
    # 按拓扑分析
    print("\n按拓扑分析最佳算法:")
    print("-" * 40)
    for topology in sorted(topology_stats.keys()):
        results = topology_stats[topology]
        avg_score = sum(r['score'] for r in results) / len(results)
        algorithms = [r['algorithm'] for r in results]
        most_common = max(set(algorithms), key=algorithms.count)
        print(f"{topology:12}: 平均分数 {avg_score:6.3f}, 最常获胜: {most_common}")
    
    # 按控制器数量分析
    print("\n按控制器数量分析:")
    print("-" * 30)
    for controller_num in sorted(controller_stats.keys()):
        results = controller_stats[controller_num]
        avg_score = sum(r['score'] for r in results) / len(results)
        algorithms = [r['algorithm'] for r in results]
        most_common = max(set(algorithms), key=algorithms.count)
        print(f"控制器数量 {controller_num}: 平均分数 {avg_score:6.3f}, 最常获胜: {most_common}")
    
    # 总体最佳算法
    print("\n" + "=" * 50)
    best_overall = max(algorithm_wins.items(), key=lambda x: x[1])
    print(f"总体表现最佳算法: {best_overall[0]} (获胜 {best_overall[1]} 次)")
    print("=" * 50)

def main():
    """
    主函数
    """
    file_path = 'test_data.txt'
    
    try:
        print("正在解析测试数据...")
        results = parse_test_data(file_path)
        print(f"成功解析 {len(results)} 条测试结果")
        
        print("\n正在分析算法性能...")
        best_algorithms, all_scores = analyze_algorithms(results)
        
        print("\n正在生成汇总报告...")
        summarize_results(best_algorithms)
        
        # 保存详细结果到JSON文件
        output_data = {
            'best_algorithms': {
                f"{k[0]}_{k[1]}": v for k, v in best_algorithms.items()
            },
            'summary': {
                'total_configs': len(best_algorithms),
                'avg_score': sum(all_scores) / len(all_scores) if all_scores else 0,
                'min_score': min(all_scores) if all_scores else 0,
                'max_score': max(all_scores) if all_scores else 0
            }
        }
        
        with open('algorithm_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print("\n详细结果已保存到 algorithm_analysis_results.json")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == '__main__':
    main()