"""
完整示例：从人设定义到OOC检测的完整流程
演示如何使用character-ooc-detector系统
"""

import sys
import os

# 获取脚本所在目录和项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 切换到项目根目录
os.chdir(PROJECT_ROOT)

# 添加src路径
sys.path.insert(0, PROJECT_ROOT)

from src.persona import Persona
from src.tag_extractor import TagExtractor
from src.sample_generator import SampleGenerator
from src.trainer import RMTrainer
from src.scorer import OOCScorer


def demo_1_persona_creation():
    """示例1: 创建和管理Persona"""
    print("=" * 60)
    print("示例1: 创建和管理角色人设")
    print("=" * 60)
    
    # 方式1: 从JSON文件加载
    print("\n1. 从JSON加载人设...")
    persona = Persona.from_json("../config/example_persona.json")
    print(persona)
    
    # 方式2: 从自然语言提取（需要OpenAI API）
    print("\n2. 从自然语言提取人设...")
    description = """
    王小明是一个现代大学生，计算机专业，性格内向但技术能力强。
    说话比较直接，经常使用网络用语和技术术语。
    重视效率和逻辑，不太擅长人际交往。
    """
    
    # 注意：这需要设置OPENAI_API_KEY环境变量
    try:
        extractor = TagExtractor()
        persona_dict = extractor.extract_tags_from_text(description)
        print("\n提取的人设结构:")
        print(f"  名称: {persona_dict['name']}")
        print(f"  维度数: {len(persona_dict['tags'])}")
    except Exception as e:
        print(f"  (跳过LLM提取，需要OpenAI API: {e})")
    
    print("\n✓ Persona创建完成")


def demo_2_sample_generation():
    """示例2: 生成训练样本"""
    print("\n" + "=" * 60)
    print("示例2: 生成偏好对训练样本")
    print("=" * 60)
    
    # 加载人设
    persona = Persona.from_json("../config/example_persona.json")
    
    print(f"\n为角色 '{persona.name}' 生成训练样本...")
    
    # 创建生成器（需要OpenAI API）
    try:
        generator = SampleGenerator(persona)
        
        # 生成少量样本作为演示
        print("生成10个偏好对...")
        pairs = generator.generate_preference_pairs(num_pairs=10)
        
        # 显示第一个样本
        if pairs:
            pair = pairs[0]
            print(f"\n示例偏好对:")
            print(f"  上下文: {pair.context}")
            print(f"  正例: {pair.chosen}")
            print(f"  负例: {pair.rejected}")
            print(f"  违背维度: {pair.violated_dimension}")
        
        # 保存到文件
        os.makedirs("../data", exist_ok=True)
        generator.save_pairs_to_jsonl(pairs, "../data/demo_pairs.jsonl")
        
        print("\n✓ 样本生成完成")
        
    except Exception as e:
        print(f"  (跳过样本生成，需要OpenAI API: {e})")


def demo_3_model_training():
    """示例3: 训练Reward Model"""
    print("\n" + "=" * 60)
    print("示例3: 训练Reward Model")
    print("=" * 60)
    
    from src.sample_generator import PreferencePair
    
    # 检查是否有训练数据
    data_path = "../data/demo_pairs.jsonl"
    if not os.path.exists(data_path):
        print("\n创建演示数据...")
        # 创建一些虚拟数据用于演示
        demo_pairs = [
            PreferencePair(
                context="师弟问：师兄最近怎么样？",
                chosen="还行，在练新招式。师傅说我进步挺快。",
                rejected="超级好啊！我在学Python编程，特别有意思！",
                violated_dimension="knowledge"
            ),
            PreferencePair(
                context="有人背叛了师门",
                chosen="背信弃义！我定要讨个说法。",
                rejected="算了算了，大家和气生财嘛。",
                violated_dimension="values"
            )
        ] * 25  # 复制以达到50个样本
        
    else:
        # 加载真实数据
        demo_pairs = SampleGenerator.load_pairs_from_jsonl(data_path)
    
    print(f"\n加载了 {len(demo_pairs)} 个训练样本")
    
    # 创建训练器
    print("\n初始化训练器...")
    trainer = RMTrainer(
        model_name="microsoft/deberta-v3-base",
        output_dir="../checkpoints/demo"
    )
    
    # 训练（使用小参数快速演示）
    print("\n开始训练（演示版本，仅1个epoch）...")
    print("注意：完整训练建议使用更多数据和更多epoch")
    
    trainer.train(
        train_pairs=demo_pairs[:40],
        val_pairs=demo_pairs[40:50],
        num_epochs=1,
        batch_size=4,
        learning_rate=2e-5,
        save_steps=20,
        eval_steps=20
    )
    
    print("\n✓ 模型训练完成")


def demo_4_ooc_detection():
    """示例4: 使用训练好的模型检测OOC"""
    print("\n" + "=" * 60)
    print("示例4: OOC检测")
    print("=" * 60)
    
    model_path = "../checkpoints/demo/best_model"
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"\n模型未找到: {model_path}")
        print("请先运行demo_3_model_training()训练模型")
        return
    
    # 加载评分器
    print(f"\n加载模型: {model_path}")
    scorer = OOCScorer(model_path=model_path, threshold=0.0)
    
    # 加载人设
    persona = Persona.from_json("../config/example_persona.json")
    
    # 测试场景
    test_cases = [
        {
            "context": "李四问：'师兄，最近修炼得如何？'",
            "responses": [
                "还行，这套拳法颇有心得。",  # 符合人设
                "嗨呀，我在刷抖音学习新的健身方法！",  # 违背knowledge
                "挺好的呀，你也要加油哦！爱你么么哒~",  # 违背style
            ]
        },
        {
            "context": "有人背叛了师门",
            "responses": [
                "背信弃义之人，我必不轻饶。",  # 符合人设
                "算了，大家都不容易，原谅他吧。",  # 违背values
            ]
        }
    ]
    
    # 测试每个场景
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"测试场景 {i}: {test['context']}")
        print('='*50)
        
        for j, response in enumerate(test['responses'], 1):
            print(f"\n回复 {j}: {response}")
            result = scorer.score(
                context=test['context'],
                response=response,
                persona=persona
            )
            print(scorer.format_result(result, verbose=False))
    
    # 比较多个回复
    print(f"\n{'='*50}")
    print("回复比较示例")
    print('='*50)
    
    comparison = scorer.compare_responses(
        context=test_cases[0]['context'],
        responses=test_cases[0]['responses'],
        labels=["符合人设", "违背知识", "违背风格"]
    )
    
    print(f"\n上下文: {comparison['context']}")
    print(f"\n排名:")
    for rank, idx in enumerate(comparison['ranked_indices'], 1):
        print(f"  {rank}. {comparison['labels'][idx]}: "
              f"{comparison['responses'][idx][:30]}... "
              f"(分数: {comparison['scores'][idx]['overall_score']:.3f})")
    
    print("\n✓ OOC检测完成")


def demo_5_violation_analysis():
    """示例5: 详细违背分析"""
    print("\n" + "=" * 60)
    print("示例5: 详细违背分析")
    print("=" * 60)
    
    model_path = "../checkpoints/demo/best_model"
    
    if not os.path.exists(model_path):
        print(f"\n模型未找到，请先运行训练")
        return
    
    scorer = OOCScorer(model_path=model_path)
    persona = Persona.from_json("../config/example_persona.json")
    
    # 分析一个明显违背的回复
    context = "师父问：'近来修行如何？'"
    bad_response = "超棒的！我在用AI学习量子物理，还在ins上分享健身视频呢！"
    
    print(f"\n上下文: {context}")
    print(f"回复: {bad_response}")
    
    analysis = scorer.analyze_violations(
        context=context,
        response=bad_response,
        persona=persona,
        threshold_per_dim=-0.5
    )
    
    print(f"\n是否OOC: {'是' if analysis['is_ooc'] else '否'}")
    print(f"违背数量: {analysis['num_violations']}")
    
    if analysis['violations']:
        print(f"\n具体违背:")
        for v in analysis['violations']:
            print(f"\n  维度: {v['dimension']}")
            print(f"  分数: {v['score']:.3f}")
            print(f"  期望: {v['expected']}")
            if v['examples']:
                print(f"  示例: {v['examples'][0]}")
    
    print("\n✓ 违背分析完成")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Character OOC Detector - 完整示例")
    print("=" * 60)
    
    demos = [
        ("Persona创建", demo_1_persona_creation),
        ("样本生成", demo_2_sample_generation),
        ("模型训练", demo_3_model_training),
        ("OOC检测", demo_4_ooc_detection),
        ("违背分析", demo_5_violation_analysis),
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  0. 运行所有示例")
    
    try:
        choice = input("\n请选择要运行的示例 (0-5): ").strip()
        choice = int(choice)
        
        if choice == 0:
            for name, func in demos:
                try:
                    func()
                except Exception as e:
                    print(f"\n[错误] {name} 失败: {e}")
                    import traceback
                    traceback.print_exc()
        elif 1 <= choice <= len(demos):
            demos[choice - 1][1]()
        else:
            print("无效选择")
    
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()