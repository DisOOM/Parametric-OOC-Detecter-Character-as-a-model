#!/usr/bin/env python3
"""
快速开始脚本
演示Character OOC Detector的基本功能
"""

import os
import sys

# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 切换到项目根目录
os.chdir(SCRIPT_DIR)

def print_banner():
    """打印欢迎信息"""
    print("=" * 70)
    print("  Character OOC Detector - 角色人格一致性检测器")
    print("  MVP版本 - 快速开始向导")
    print("=" * 70)
    print()
    print("📌 数据生成策略:")
    print("  1. 优先尝试使用OpenAI API生成高质量训练数据")
    print("  2. 生成的数据会缓存到 data/ 目录，可重复使用")
    print("  3. 如API未配置，自动降级使用fallback演示数据")
    print("  4. fallback数据质量较低但足够体验完整流程")
    print()
    print("💡 提示: 配置API可获得更好的训练效果")
    print("   参考: CONFIG_GUIDE.md 或 config/config.yaml.example")
    print()

def check_dependencies():
    """检查依赖是否安装"""
    print("📦 检查依赖...")
    
    required_packages = [
        "torch",
        "transformers",
        "datasets"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package} (未安装)")
    
    if missing:
        print(f"\n❌ 缺少依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖已安装\n")
    return True

def demo_persona():
    """演示Persona功能"""
    print("\n" + "=" * 70)
    print("📋 步骤 1: 选择并加载角色人设")
    print("=" * 70)
    
    from src import Persona
    
    # 列出可用的人设
    personas = {
        "1": {
            "name": "张三 - 江湖侠客",
            "path": "config/example_persona.json",
            "description": "少林弟子，重情重义，说话简洁直接"
        },
        "2": {
            "name": "艾莉娅 - 猫娘女仆",
            "path": "config/catgirl_maid_persona.json",
            "description": "维多利亚时代女仆，温柔礼貌，带猫系口癖"
        }
    }
    
    print("\n可选角色人设：")
    for key, info in personas.items():
        print(f"  {key}. {info['name']}")
        print(f"     {info['description']}")
    
    # 用户选择
    choice = input("\n请选择人设 (1/2，默认1): ").strip() or "1"
    
    if choice not in personas:
        print(f"无效选择，使用默认人设")
        choice = "1"
    
    selected = personas[choice]
    persona_path = selected["path"]
    
    if not os.path.exists(persona_path):
        print(f"❌ 人设文件不存在: {persona_path}")
        return None, None
    
    persona = Persona.from_json(persona_path)
    print(f"\n✅ 成功加载角色: {selected['name']}")
    print(persona)
    
    # 验证完整性
    is_valid = persona.validate()
    if is_valid:
        print("\n✓ 人设验证通过")
    
    return persona, choice

def demo_training(persona_choice="1"):
    """演示训练流程"""
    print("\n" + "=" * 70)
    print("🔧 步骤 2: 准备训练数据")
    print("=" * 70)
    
    from src import RMTrainer, PreferencePair, SampleGenerator, Persona
    from src.config_loader import get_config
    
    # 确定数据文件路径
    persona_names = {"1": "zhangsan", "2": "catgirl"}
    persona_paths = {"1": "config/example_persona.json", "2": "config/catgirl_maid_persona.json"}
    
    data_file = f"data/training_pairs_{persona_names[persona_choice]}.jsonl"
    
    # 检查是否已有缓存数据
    if os.path.exists(data_file):
        existing_pairs = SampleGenerator.load_pairs_from_jsonl(data_file)
        print(f"\n✓ 发现缓存数据: {data_file} ({len(existing_pairs)} 个样本)")
        
        print("\n选择操作:")
        print("  1. 使用现有数据")
        print("  2. 追加新数据到现有文件")
        print("  3. 重新生成（覆盖）")
        
        choice = input("\n请选择 (1/2/3，默认1): ").strip() or "1"
        
        if choice == "1":
            print("使用现有数据...")
            demo_pairs = existing_pairs
        elif choice == "2":
            print("追加模式...")
            new_pairs = _generate_training_data(persona_choice, persona_paths, data_file, append_mode=True)
            if new_pairs:
                demo_pairs = existing_pairs + new_pairs
                print(f"✓ 总计: {len(demo_pairs)} 个样本 (原有{len(existing_pairs)} + 新增{len(new_pairs)})")
            else:
                demo_pairs = existing_pairs
        else:
            print("重新生成模式...")
            demo_pairs = _generate_training_data(persona_choice, persona_paths, data_file, append_mode=False)
    else:
        demo_pairs = _generate_training_data(persona_choice, persona_paths, data_file, append_mode=False)
    
    if not demo_pairs:
        print("❌ 无法获取训练数据")
        return None
    
    # 继续训练流程...
    _run_training(demo_pairs)
    
    return "./checkpoints/quick_start"


def _generate_training_data(persona_choice, persona_paths, data_file, append_mode=False):
    """生成或使用fallback训练数据"""
    from src import SampleGenerator, Persona
    from src.config_loader import get_config
    
    print("\n尝试使用LLM生成训练数据...")
    
    # 检查API配置
    config = get_config()
    if config.check_api_configured():
        try:
            print("✓ OpenAI API已配置，生成高质量训练数据...")
            persona = Persona.from_json(persona_paths[persona_choice])
            generator = SampleGenerator(persona)
            
            # 从配置读取生成数量
            sample_config = config.get_sample_generation_config()
            default_num = sample_config.get("num_pairs", 100)
            
            # 计算起始索引（追加模式下避免index重复）
            start_index = 0
            if append_mode and os.path.exists(data_file):
                existing = SampleGenerator.load_pairs_from_jsonl(data_file)
                start_index = len(existing)
            
            # 如果是追加模式，询问数量
            if append_mode:
                print(f"\n当前配置默认生成: {default_num} 个样本")
                num_input = input(f"要生成多少个新样本？(默认{default_num}): ").strip()
                num_pairs = int(num_input) if num_input else default_num
            else:
                num_pairs = default_num
                print(f"生成 {num_pairs} 个偏好对...")
            
            # 生成偏好对（使用正确的起始索引）
            demo_pairs = generator.generate_preference_pairs(
                num_pairs=num_pairs,
                start_index=start_index
            )
            
            # 保存到文件
            os.makedirs("data", exist_ok=True)
            
            if append_mode:
                # 追加模式：先读取现有数据
                existing = []
                if os.path.exists(data_file):
                    existing = SampleGenerator.load_pairs_from_jsonl(data_file)
                
                # 合并
                all_pairs = existing + demo_pairs
                generator.save_pairs_to_jsonl(all_pairs, data_file)
                print(f"✓ 已追加 {num_pairs} 个样本到 {data_file}")
                print(f"✓ 文件现有 {len(all_pairs)} 个样本")
                return demo_pairs  # 只返回新生成的
            else:
                # 覆盖模式
                generator.save_pairs_to_jsonl(demo_pairs, data_file)
                print(f"✓ 已保存 {num_pairs} 个样本到 {data_file}")
                return demo_pairs
            
        except Exception as e:
            print(f"⚠️  LLM生成失败: {e}")
            print("将使用fallback演示数据...")
            return _create_fallback_data(persona_choice)
    else:
        print("⚠️  OpenAI API未配置")
        config.print_config_guide()
        print("\n将使用fallback演示数据（质量较低，仅供快速测试）")
        
        choice = input("\n是否继续使用fallback数据？(y/n): ").strip().lower()
        if choice != 'y':
            return None
        
        return _create_fallback_data(persona_choice)


def _create_fallback_data(persona_choice):
    """创建fallback演示数据（硬编码）"""
    from src import PreferencePair
    
    print("\n使用硬编码的演示数据...")
    
    if persona_choice == "2":
        # 猫娘女仆的fallback数据
        demo_pairs = [
            PreferencePair(
                context="主人回到家中",
                chosen="欢迎回家，主人~艾莉娅已经准备好晚餐了喵！",
                rejected="哟，回来啦？晚饭在桌上，自己吃吧。",
                violated_dimension="style"
            ),
            PreferencePair(
                context="主人问：'今天天气怎么样？'",
                chosen="今天是个晴朗的好天气呢，主人要出门吗喵？",
                rejected="我看了天气预报，今天多云转晴，气温25度。",
                violated_dimension="knowledge"
            ),
            PreferencePair(
                context="主人身体不适",
                chosen="主人！艾莉娅这就去准备热水和毛巾，请您好好休息喵~",
                rejected="那你自己去看医生吧，我还有其他事要忙。",
                violated_dimension="value_system"
            ),
            PreferencePair(
                context="客人来访",
                chosen="老爷，主人正在书房，艾莉娅这就去通知~",
                rejected="嘿，那个谁，等着啊，我去叫人。",
                violated_dimension="etiquette"
            ),
            PreferencePair(
                context="主人遇到危险",
                chosen="主人！请让艾莉娅来保护您喵！请您站在艾莉娅身后！",
                rejected="哎呀，主人你自己小心点啊，我去躲一下。",
                violated_dimension="stress"
            )
        ] * 10  # 复制以达到50个样本
    else:
        # 江湖侠客的fallback数据
        demo_pairs = [
            PreferencePair(
                context="李四问：'师兄最近怎么样？'",
                chosen="还行，在练新招式。",
                rejected="超级好！我在学Python编程呢！",
                violated_dimension="knowledge"
            ),
            PreferencePair(
                context="有人背叛了师门",
                chosen="背信弃义之人，我必不轻饶。",
                rejected="算了，大家都不容易，原谅吧。",
                violated_dimension="value_system"
            ),
            PreferencePair(
                context="师父问：'修炼得如何？'",
                chosen="弟子不敢懈怠，日日苦练。",
                rejected="嘿嘿，还不错啦，老师你也要加油哦！",
                violated_dimension="etiquette"
            )
        ] * 17  # 复制以达到51个样本
    
    print(f"✓ 创建了 {len(demo_pairs)} 个fallback样本（数据重复，仅供演示）")
    return demo_pairs


def _run_training(demo_pairs):
    """执行训练"""
    from src import RMTrainer
    
    print(f"\n准备训练 {len(demo_pairs)} 个样本")
    
    # 选择模型大小
    print("\n" + "─" * 70)
    print("📊 选择预训练模型大小:")
    print("─" * 70)
    
    models = {
        "0": ("OpenAssistant/reward-model-deberta-v3-base", "184M", "🔥 推荐！从预训练RM微调，效果更好更稳定"),
        "1": ("microsoft/deberta-v3-small", "44M", "从头训练，轻量快速，适合CPU或快速测试"),
        "2": ("microsoft/deberta-v3-base", "184M", "从头训练，平衡性能"),
        "3": ("microsoft/deberta-v3-large", "434M", "从头训练，高性能，需要GPU和大显存"),
        "4": ("roberta-base", "125M", "经典选择 (从头训练)"),
        "5": ("roberta-large", "355M", "大规模模型 (从头训练)")
    }
    
    print("\n可选模型:")
    for key, (name, size, desc) in models.items():
        print(f"  {key}. {name}")
        print(f"     参数量: {size} | {desc}")
    
    model_choice = input("\n请选择模型 (0-5，默认0): ").strip() or "0"
    
    if model_choice not in models:
        print("无效选择，使用默认模型")
        model_choice = "0"

    selected_model_path = models[model_choice][0]
    
    # 区分是微调RM还是从头训练
    pretrained_rm_path = None
    base_model_name_for_tokenizer = "microsoft/deberta-v3-base" # 默认

    if model_choice == "0":
        # 微调预训练的RM
        pretrained_rm_path = selected_model_path
        # 需要指定其基础模型，以便加载正确的tokenizer
        base_model_name_for_tokenizer = "microsoft/deberta-v3-base"
        print(f"\n✓ 选择微调预训练RM: {pretrained_rm_path} ({models[model_choice][1]})")
    else:
        # 从头训练
        base_model_name_for_tokenizer = selected_model_path
        print(f"\n✓ 选择模型从头训练: {base_model_name_for_tokenizer} ({models[model_choice][1]})")

    # 创建训练器
    print("\n初始化训练器...")
    output_dir = "./checkpoints/quick_start"
    
    trainer = RMTrainer(
        model_name=base_model_name_for_tokenizer,
        output_dir=output_dir,
        pretrained_rm_path=pretrained_rm_path
    )
    
    # 询问是否要训练
    print("\n⚠️  训练将下载预训练模型（约400MB）并需要一些时间")
    print("   - CPU训练: 约10-20分钟")
    print("   - GPU训练: 约2-5分钟")
    
    choice = input("\n是否继续训练? (y/n): ").strip().lower()
    
    if choice != 'y':
        print("跳过训练步骤")
        return None
    
    # 智能划分训练集和验证集（80/20）
    total_samples = len(demo_pairs)
    train_split = int(total_samples * 0.8)
    
    train_pairs = demo_pairs[:train_split]
    val_pairs = demo_pairs[train_split:]
    
    print(f"\n📊 数据划分:")
    print(f"  训练集: {len(train_pairs)} 个样本 ({len(train_pairs)/total_samples*100:.0f}%)")
    print(f"  验证集: {len(val_pairs)} 个样本 ({len(val_pairs)/total_samples*100:.0f}%)")
    
    # 根据数据量调整训练参数
    if total_samples < 100:
        num_epochs = 2
        save_steps = max(10, train_split // 4)
    else:
        num_epochs = 4
        save_steps = max(25, train_split // 4)
    
    eval_steps = save_steps
    
    print(f"\n⚙️  训练配置:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: 4")
    print(f"  保存/评估间隔: 每 {save_steps} steps")
    
    # 开始训练
    print("\n开始训练...")
    trainer.train(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        num_epochs=num_epochs,
        batch_size=2,
        learning_rate=1.6e-5,
        save_steps=save_steps,
        eval_steps=eval_steps,
        use_bf16=False  # 使用BF16节省显存
    )
    
    print(f"\n✅ 模型训练完成，保存在: {output_dir}")

def demo_scoring(model_path, persona_choice="1"):
    """演示评分功能"""
    print("\n" + "=" * 70)
    print("🎯 步骤 3: 使用模型进行OOC检测")
    print("=" * 70)
    
    from src import OOCScorer, Persona
    
    # 根据选择加载对应人设
    if persona_choice == "2":
        persona_path = "config/catgirl_maid_persona.json"
        print(f"\n使用角色: 艾莉娅 - 猫娘女仆")
    else:
        persona_path = "config/example_persona.json"
        print(f"\n使用角色: 张三 - 江湖侠客")
    
    persona = Persona.from_json(persona_path)
    
    # 检查模型是否存在
    if not model_path or not os.path.exists(model_path):
        print(f"\n❌ 模型不存在: {model_path}")
        print("请先完成训练步骤")
        return
    
    # 加载评分器
    print(f"\n加载模型: {model_path}")
    scorer = OOCScorer(model_path=model_path)
    
    # 根据角色准备不同的测试样本
    if persona_choice == "2":
        # 猫娘女仆的测试样本
        test_cases = [
            {
                "context": "主人回到家中",
                "response": "欢迎回家，主人~艾莉娅已经准备好晚餐了喵！",
                "label": "✅ 符合人设"
            },
            {
                "context": "主人回到家中",
                "response": "哟，回来了？我正在看手机呢，晚饭你自己热一下。",
                "label": "❌ 违背风格+知识+价值观"
            },
            {
                "context": "主人问：'能帮我泡杯茶吗？'",
                "response": "当然，艾莉娅这就去准备红茶呢，请主人稍候喵~",
                "label": "✅ 符合人设"
            },
            {
                "context": "主人问：'能帮我泡杯茶吗？'",
                "response": "好的，我去用咖啡机给你冲一杯速溶咖啡。",
                "label": "❌ 违背知识边界"
            },
            {
                "context": "主人身体不适",
                "response": "主人！艾莉娅这就去准备热水和毛巾，请您好好休息喵~",
                "label": "✅ 符合人设"
            },
            {
                "context": "主人身体不适",
                "response": "那你自己去看医生吧，我还有事要做。",
                "label": "❌ 违背价值观"
            }
        ]
    else:
        # 江湖侠客的测试样本
        test_cases = [
            {
                "context": "李四问：'师兄，最近修炼如何？'",
                "response": "还行，这套拳法颇有心得。",
                "label": "✅ 符合人设"
            },
            {
                "context": "李四问：'师兄，最近修炼如何？'",
                "response": "我在用AI学习量子计算，特别酷！",
                "label": "❌ 违背知识边界"
            },
            {
                "context": "有人背叛了师门",
                "response": "背信弃义，我定要讨个说法。",
                "label": "✅ 符合人设"
            },
            {
                "context": "有人背叛了师门",
                "response": "算了算了，和气生财嘛~",
                "label": "❌ 违背价值观"
            }
        ]
    
    print(f"\n测试 {len(test_cases)} 个样本:\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"{'─' * 70}")
        print(f"测试 {i}: {test['label']}")
        print(f"{'─' * 70}")
        
        result = scorer.score(
            context=test['context'],
            response=test['response'],
            persona=persona
        )
        
        print(scorer.format_result(result, verbose=True))
        print()
    
    print("✅ OOC检测演示完成")


def demo_custom_test(model_path, persona_choice="1"):
    """自定义测试"""
    print("\n" + "=" * 70)
    print("🧪 步骤 4: 自定义测试")
    print("=" * 70)
    
    from src import OOCScorer, Persona
    
    # 加载人设和模型
    if persona_choice == "2":
        persona_path = "config/catgirl_maid_persona.json"
    else:
        persona_path = "config/example_persona.json"
    
    persona = Persona.from_json(persona_path)
    scorer = OOCScorer(model_path=model_path)
    
    print(f"\n当前角色: {persona.name}")
    print("\n你可以输入自定义的对话场景来测试OOC检测")
    print("输入 'q' 退出自定义测试\n")
    
    while True:
        print("─" * 70)
        context = input("📝 输入对话场景/上下文 (或输入q退出): ").strip()
        
        if context.lower() == 'q':
            print("\n退出自定义测试")
            break
        
        if not context:
            print("⚠️  上下文不能为空")
            continue
        
        # 可以输入多个候选回复进行比较
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
                print(f"\n  {rank}. {status} 回复: {responses[idx]}")
                print(f"     总分: {score:.3f}")
                
                # 显示各维度分
                dim_scores = comparison['scores'][idx]['dimension_scores']
                print("     维度:", end="")
                for dim, s in dim_scores.items():
                    print(f" {dim[:3]}:{s:+.2f}", end="")
                print()
        
        print()


def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    try:
        # 步骤1: 加载人设
        persona, persona_choice = demo_persona()
        if not persona:
            print("\n❌ 无法加载人设")
            sys.exit(1)
        
        input("\n按回车继续...")
        
        # 步骤2: 训练模型
        model_path = demo_training(persona_choice)
        
        # 如果训练成功，继续评分演示
        if model_path:
            input("\n按回车继续...")
            
            # 步骤3: 预设测试
            best_model = os.path.join(model_path, "best_model")
            final_model = os.path.join(model_path, "final_model")
            
            active_model = None
            if os.path.exists(best_model):
                active_model = best_model
                demo_scoring(best_model, persona_choice)
            elif os.path.exists(final_model):
                active_model = final_model
                demo_scoring(final_model, persona_choice)
            else:
                print("\n⚠️  未找到训练好的模型")
            
            # 步骤4: 自定义测试
            if active_model:
                input("\n按回车继续进入自定义测试...")
                demo_custom_test(active_model, persona_choice)
        
        # 完成
        print("\n" + "=" * 70)
        print("🎉 快速开始演示完成！")
        print("=" * 70)
        print("\n下一步:")
        print("  1. 查看完整示例: python examples/demo.py")
        print("  2. 阅读文档: README.md")
        print("  3. 创建自己的角色人设")
        print("  4. 生成更多训练数据（需要OpenAI API）")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()