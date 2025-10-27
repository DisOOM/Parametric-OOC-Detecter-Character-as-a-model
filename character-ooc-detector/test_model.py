#!/usr/bin/env python3
"""
模型测试工具
加载已训练的模型进行OOC检测测试
"""

import os
import sys
import glob

# 切换到项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from src import OOCScorer, Persona


def list_available_models():
    """列出可用的模型"""
    checkpoint_dir = "checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ 模型目录不存在: {checkpoint_dir}")
        return []
    
    models = []
    
    # 查找所有包含训练信息的模型目录
    for root, dirs, files in os.walk(checkpoint_dir):
        if "training_info.json" in files and "pytorch_model.bin" in files:
            rel_path = os.path.relpath(root, ".")
            models.append(rel_path)
    
    return models


def list_available_personas():
    """列出可用的人设"""
    personas = {}
    config_dir = "config"
    
    for f in os.listdir(config_dir):
        if f.endswith("_persona.json"):
            path = os.path.join(config_dir, f)
            try:
                persona = Persona.from_json(path)
                personas[persona.name] = path
            except:
                pass
    
    return personas


def select_model():
    """选择模型"""
    print("\n" + "=" * 70)
    print("🤖 选择模型")
    print("=" * 70)
    
    models = list_available_models()
    
    if not models:
        print("\n❌ 没有找到训练好的模型")
        print("请先运行 quick_start.py 训练模型，或使用 examples/demo.py")
        return None
    
    print(f"\n找到 {len(models)} 个模型:")
    for i, model_path in enumerate(models, 1):
        print(f"  {i}. {model_path}")
    
    choice = input(f"\n选择模型 (1-{len(models)}): ").strip()
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
    except ValueError:
        pass
    
    print("无效选择")
    return None


def select_persona():
    """选择人设"""
    print("\n" + "=" * 70)
    print("👤 选择角色人设")
    print("=" * 70)
    
    personas = list_available_personas()
    
    if not personas:
        print("\n❌ 没有找到人设文件")
        return None, None
    
    print(f"\n可用角色:")
    names = list(personas.keys())
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    
    choice = input(f"\n选择角色 (1-{len(names)}, 或直接输入角色名): ").strip()
    
    # 尝试作为数字
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(names):
            name = names[idx]
            return Persona.from_json(personas[name]), name
    except ValueError:
        # 尝试作为名字
        if choice in personas:
            return Persona.from_json(personas[choice]), choice
    
    print("无效选择")
    return None, None


def batch_test_mode(scorer, persona):
    """批量测试模式"""
    print("\n" + "=" * 70)
    print("📝 批量测试模式")
    print("=" * 70)
    print("\n从文件读取测试用例（每行: 上下文|回复）")
    print("示例文件格式:")
    print("  主人回到家|欢迎回家，主人~")
    print("  主人回到家|哟，回来了？")
    
    file_path = input("\n输入测试文件路径 (或按回车跳过): ").strip()
    
    if not file_path or not os.path.exists(file_path):
        print("跳过批量测试")
        return
    
    # 读取测试用例
    test_cases = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                context, response = line.split('|', 1)
                test_cases.append((context.strip(), response.strip()))
    
    if not test_cases:
        print("❌ 没有找到有效的测试用例")
        return
    
    print(f"\n找到 {len(test_cases)} 个测试用例")
    
    # 批量评分
    contexts = [t[0] for t in test_cases]
    responses = [t[1] for t in test_cases]
    
    results = scorer.score_batch(contexts, responses)
    
    # 显示结果
    print("\n" + "=" * 70)
    print("批量测试结果:")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        status = "✓" if result['passed'] else "✗"
        print(f"\n{i}. {status} {result['context']}")
        print(f"   回复: {result['response']}")
        print(f"   总分: {result['overall_score']:.3f}")
        print(f"   最弱维度: {result['weakest_dimension']} ({result['weakest_score']:.3f})")


def interactive_test_mode(scorer, persona):
    """交互式测试模式"""
    print("\n" + "=" * 70)
    print("🧪 交互式测试模式")
    print("=" * 70)
    print(f"\n当前角色: {persona.name}")
    print("输入 'q' 退出测试\n")
    
    while True:
        print("─" * 70)
        context = input("📝 输入对话场景/上下文 (或q退出): ").strip()
        
        if context.lower() == 'q':
            break
        
        if not context:
            print("⚠️  上下文不能为空")
            continue
        
        # 输入回复
        responses = []
        print("\n输入回复（至少1个，输入空行结束）:")
        
        while True:
            response = input(f"  回复 {len(responses) + 1}: ").strip()
            if not response:
                break
            responses.append(response)
        
        if not responses:
            print("⚠️  至少需要一个回复")
            continue
        
        # 评分
        print(f"\n{'='*70}")
        print("评分结果:")
        print('='*70)
        
        if len(responses) == 1:
            # 单个回复
            result = scorer.score(
                context=context,
                response=responses[0],
                persona=persona
            )
            print(scorer.format_result(result, verbose=True))
        else:
            # 多个回复比较
            comparison = scorer.compare_responses(
                context=context,
                responses=responses
            )
            
            print(f"\n上下文: {context}\n")
            print("排名:")
            for rank, idx in enumerate(comparison['ranked_indices'], 1):
                score = comparison['scores'][idx]['overall_score']
                status = "✓" if comparison['scores'][idx]['passed'] else "✗"
                print(f"\n  {rank}. {status} {responses[idx]}")
                print(f"     总分: {score:.3f}")
                
                # 显示各维度分
                dim_scores = comparison['scores'][idx]['dimension_scores']
                print("     维度:", end="")
                for dim, s in dim_scores.items():
                    print(f" {dim[:4]}:{s:+.2f}", end="")
                print()
        
        print()


def preset_test_mode(scorer, persona):
    """预设测试模式"""
    print("\n" + "=" * 70)
    print("📋 预设测试场景")
    print("=" * 70)
    
    # 根据角色准备测试
    if "艾莉娅" in persona.name or "女仆" in str(persona.tags):
        # 猫娘女仆测试
        test_cases = [
            ("主人回家", [
                "欢迎回家，主人~艾莉娅已经准备好晚餐了喵！",
                "哟，回来了？晚饭自己吃吧。"
            ]),
            ("主人问：能帮我泡茶吗？", [
                "当然，艾莉娅这就去准备红茶呢，请主人稍候喵~",
                "好的，我用咖啡机给你冲咖啡。"
            ]),
            ("主人身体不适", [
                "主人！艾莉娅这就去准备热水和毛巾喵~",
                "那你自己去看医生吧。"
            ])
        ]
    else:
        # 江湖侠客测试
        test_cases = [
            ("李四问：'师兄最近如何？'", [
                "还行，这套拳法颇有心得。",
                "我在用AI学习量子计算！"
            ]),
            ("有人背叛了师门", [
                "背信弃义，我定要讨个说法。",
                "算了算了，和气生财嘛~"
            ])
        ]
    
    print(f"\n为角色 '{persona.name}' 准备了 {len(test_cases)} 个测试场景\n")
    
    for i, (context, responses) in enumerate(test_cases, 1):
        print("─" * 70)
        print(f"场景 {i}: {context}")
        
        comparison = scorer.compare_responses(
            context=context,
            responses=responses
        )
        
        print("\n排名:")
        for rank, idx in enumerate(comparison['ranked_indices'], 1):
            score = comparison['scores'][idx]['overall_score']
            status = "✓" if comparison['scores'][idx]['passed'] else "✗"
            response_preview = responses[idx][:40] + "..." if len(responses[idx]) > 40 else responses[idx]
            print(f"  {rank}. {status} {response_preview}")
            print(f"     分数: {score:.3f}")
        print()


def main():
    """主函数"""
    print("=" * 70)
    print("  Character OOC Detector - 模型测试工具")
    print("=" * 70)
    print()
    print("💡 说明: 此工具用于测试已训练的模型")
    print("   如需训练新模型，请使用 quick_start.py")
    print()
    
    # 选择模型
    model_path = select_model()
    if not model_path:
        sys.exit(1)
    
    print(f"\n✓ 选择的模型: {model_path}")
    
    # 选择人设
    persona, persona_name = select_persona()
    if not persona:
        sys.exit(1)
    
    print(f"✓ 选择的角色: {persona_name}")
    
    # 加载模型
    print(f"\n加载模型...")
    try:
        scorer = OOCScorer(model_path=model_path)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 测试菜单
    while True:
        print("\n" + "─" * 70)
        print("测试模式:")
        print("─" * 70)
        print("  1. 预设测试场景")
        print("  2. 交互式测试")
        print("  3. 批量测试（从文件）")
        print("  4. 切换角色人设")
        print("  5. 切换模型")
        print("  0. 退出")
        
        choice = input("\n请选择 (0-5): ").strip()
        
        if choice == "0":
            print("\n退出")
            break
        elif choice == "1":
            preset_test_mode(scorer, persona)
        elif choice == "2":
            interactive_test_mode(scorer, persona)
        elif choice == "3":
            batch_test_mode(scorer, persona)
        elif choice == "4":
            new_persona, new_name = select_persona()
            if new_persona:
                persona, persona_name = new_persona, new_name
                print(f"✓ 已切换到: {persona_name}")
        elif choice == "5":
            new_model = select_model()
            if new_model:
                try:
                    scorer = OOCScorer(model_path=new_model)
                    model_path = new_model
                    print(f"✓ 已切换到: {model_path}")
                except Exception as e:
                    print(f"❌ 加载失败: {e}")
        else:
            print("无效选择")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()