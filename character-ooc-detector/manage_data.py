#!/usr/bin/env python3
"""
数据管理工具
用于生成、追加、查看和管理训练数据
"""

import os
import sys

# 切换到项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from src import Persona, SampleGenerator
from src.config_loader import get_config


def list_personas():
    """列出可用的人设"""
    personas = {
        "1": ("张三 - 江湖侠客", "config/example_persona.json"),
        "2": ("艾莉娅 - 猫娘女仆", "config/catgirl_maid_persona.json")
    }
    
    print("\n可用角色:")
    for key, (name, path) in personas.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {key}. {exists} {name}")
    
    return personas


def view_data_stats(data_file):
    """查看数据统计"""
    if not os.path.exists(data_file):
        print(f"❌ 文件不存在: {data_file}")
        return
    
    pairs = SampleGenerator.load_pairs_from_jsonl(data_file)
    
    print(f"\n📊 数据文件: {data_file}")
    print(f"  总样本数: {len(pairs)}")
    
    # 统计各维度分布
    dim_count = {}
    for pair in pairs:
        dim = pair.violated_dimension
        dim_count[dim] = dim_count.get(dim, 0) + 1
    
    print(f"\n  维度分布:")
    for dim, count in sorted(dim_count.items()):
        percentage = count / len(pairs) * 100
        print(f"    {dim:15s}: {count:3d} ({percentage:5.1f}%)")


def generate_new_data():
    """生成新的训练数据"""
    print("\n" + "=" * 70)
    print("🎲 生成新的训练数据")
    print("=" * 70)
    
    # 选择角色
    personas = list_personas()
    choice = input("\n选择角色 (1/2): ").strip()
    
    if choice not in personas:
        print("无效选择")
        return
    
    persona_name, persona_path = personas[choice]
    persona = Persona.from_json(persona_path)
    
    # 确定数据文件
    persona_slug = {"1": "zhangsan", "2": "catgirl"}[choice]
    data_file = f"data/training_pairs_{persona_slug}.jsonl"
    
    # 检查现有数据
    existing_count = 0
    if os.path.exists(data_file):
        existing_pairs = SampleGenerator.load_pairs_from_jsonl(data_file)
        existing_count = len(existing_pairs)
        print(f"\n✓ 现有数据: {existing_count} 个样本")
    
    # 生成数量
    config = get_config()
    default_num = config.get_sample_generation_config().get("num_pairs", 100)
    
    print(f"\n配置默认数量: {default_num}")
    num_input = input(f"要生成多少个样本？(输入数量或回车使用默认): ").strip()
    num_pairs = int(num_input) if num_input else default_num
    
    if num_pairs <= 0:
        print("❌ 数量必须大于0")
        return
    
    # 追加或覆盖
    if existing_count > 0:
        print(f"\n选择操作模式:")
        print(f"  1. 追加到现有文件 (将有 {existing_count + num_pairs} 个样本)")
        print(f"  2. 覆盖现有文件 (将有 {num_pairs} 个样本)")
        
        mode_choice = input("\n请选择 (1/2，默认1): ").strip() or "1"
        append_mode = (mode_choice == "1")
    else:
        append_mode = False
    
    # 开始生成
    try:
        print(f"\n{'追加' if append_mode else '生成'} {num_pairs} 个样本...")
        print("⏳ 这可能需要一些时间，取决于数量和API速度...")
        
        generator = SampleGenerator(persona)
        
        # 计算起始索引（追加模式下避免index重复）
        start_index = 0
        if append_mode and os.path.exists(data_file):
            existing_pairs = SampleGenerator.load_pairs_from_jsonl(data_file)
            start_index = len(existing_pairs)
        
        # 生成偏好对
        new_pairs = generator.generate_preference_pairs(
            num_pairs=num_pairs,
            start_index=start_index
        )
        
        # 保存
        os.makedirs("data", exist_ok=True)
        
        if append_mode:
            existing_pairs = SampleGenerator.load_pairs_from_jsonl(data_file)
            all_pairs = existing_pairs + new_pairs
            generator.save_pairs_to_jsonl(all_pairs, data_file)
            print(f"\n✅ 成功追加 {num_pairs} 个样本")
            print(f"✅ 总计: {len(all_pairs)} 个样本 (index: {start_index}-{start_index+num_pairs-1})")
        else:
            generator.save_pairs_to_jsonl(new_pairs, data_file)
            print(f"\n✅ 成功生成 {num_pairs} 个样本 (index: 0-{num_pairs-1})")
        
        # 显示统计
        view_data_stats(data_file)
        
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()


def view_samples():
    """查看样本内容"""
    print("\n" + "=" * 70)
    print("👀 查看样本内容")
    print("=" * 70)
    
    # 列出所有数据文件
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"\n❌ 数据目录不存在: {data_dir}")
        return
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
    
    if not files:
        print(f"\n❌ 没有找到数据文件")
        return
    
    print(f"\n可用数据文件:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")
    
    choice = input(f"\n选择文件 (1-{len(files)}): ").strip()
    
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(files):
            print("无效选择")
            return
        
        data_file = os.path.join(data_dir, files[idx])
        pairs = SampleGenerator.load_pairs_from_jsonl(data_file)
        
        # 查看统计
        view_data_stats(data_file)
        
        # 查看具体样本
        print(f"\n查看前几个样本:")
        num_show = min(3, len(pairs))
        
        for i in range(num_show):
            pair = pairs[i]
            print(f"\n{'─' * 70}")
            print(f"样本 {i + 1}:")
            print(f"  上下文: {pair.context}")
            print(f"  正例: {pair.chosen}")
            print(f"  负例: {pair.rejected}")
            print(f"  违背维度: {pair.violated_dimension}")
        
        if len(pairs) > num_show:
            print(f"\n... 还有 {len(pairs) - num_show} 个样本")
    
    except (ValueError, IndexError):
        print("无效输入")


def main():
    """主函数"""
    print("=" * 70)
    print("  Character OOC Detector - 数据管理工具")
    print("=" * 70)
    
    # 检查API配置
    config = get_config()
    if not config.check_api_configured():
        print("\n⚠️  OpenAI API未配置")
        config.print_config_guide()
        
        choice = input("\n是否继续？(y/n): ").strip().lower()
        if choice != 'y':
            return
    
    while True:
        print("\n" + "─" * 70)
        print("功能菜单:")
        print("─" * 70)
        print("  1. 生成新的训练数据")
        print("  2. 查看现有数据统计")
        print("  3. 查看样本内容")
        print("  0. 退出")
        
        choice = input("\n请选择 (0-3): ").strip()
        
        if choice == "0":
            print("\n退出")
            break
        elif choice == "1":
            generate_new_data()
        elif choice == "2":
            # 列出所有数据文件
            data_dir = "data"
            if os.path.exists(data_dir):
                files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
                if files:
                    for f in files:
                        view_data_stats(os.path.join(data_dir, f))
                else:
                    print("\n❌ 没有找到数据文件")
            else:
                print("\n❌ 数据目录不存在")
        elif choice == "3":
            view_samples()
        else:
            print("\n无效选择")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()