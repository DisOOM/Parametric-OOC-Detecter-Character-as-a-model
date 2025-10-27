"""
偏好对生成器
根据Persona生成训练用的正负样本对
"""

import json
import os
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from openai import OpenAI

from .persona import Persona
from .config_loader import get_config


@dataclass
class PreferencePair:
    """偏好对数据结构"""
    context: str  # 上下文/对话场景
    chosen: str  # 正例（符合人设）
    rejected: str  # 负例（违背人设）
    violated_dimension: str  # 违背的维度
    metadata: Dict = None  # 其他元数据


class SampleGenerator:
    """样本生成器"""
    
    def __init__(self, persona: Persona, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        初始化生成器
        
        Args:
            persona: 角色人格定义
            api_key: OpenAI API密钥（优先级：参数 > 环境变量 > config.yaml）
            model: 使用的模型（如果为None，从配置读取）
        """
        self.persona = persona
        
        # 加载配置
        config = get_config()
        
        # API密钥优先级：参数 > 配置
        if api_key is None:
            api_key = config.get_openai_api_key()
        
        if not api_key:
            config.print_config_guide()
            raise ValueError("OpenAI API密钥未配置。请通过参数、环境变量或config.yaml设置")
        
        # 模型名称
        if model is None:
            model = config.get_openai_model()
        
        # 基础URL（如果配置了）
        base_url = config.get_openai_base_url()
        
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        
        self.model = model
        self.dimensions = ["style", "value_system", "knowledge", "etiquette"]
        
        print(f"✓ SampleGenerator initialized for '{persona.name}' with model: {model}")
    
    def generate_preference_pairs(
        self,
        num_pairs: int = 50,
        contexts: Optional[List[str]] = None,
        start_index: int = 0
    ) -> List[PreferencePair]:
        """
        生成偏好对
        
        Args:
            num_pairs: 需要生成的对数
            contexts: 预定义的上下文列表（可选）
            start_index: 起始索引（用于追加模式，避免重复）
            
        Returns:
            偏好对列表
        """
        pairs = []
        
        # 如果没有提供上下文，先生成
        if not contexts:
            contexts = self._generate_contexts(num_pairs)
        
        for i, context in enumerate(contexts[:num_pairs]):
            # 随机选择一个维度作为破坏目标
            dimension = random.choice(self.dimensions)
            
            # 生成正负例
            chosen, rejected = self._generate_pair_for_dimension(context, dimension)
            
            pair = PreferencePair(
                context=context,
                chosen=chosen,
                rejected=rejected,
                violated_dimension=dimension,
                metadata={"index": start_index + i, "dimension": dimension}
            )
            pairs.append(pair)
            
            # 每10个打印进度
            if (i + 1) % 10 == 0:
                print(f"已生成 {i + 1}/{num_pairs} 个偏好对")
        
        return pairs
    
    def _generate_contexts(self, num_contexts: int) -> List[str]:
        """生成对话上下文"""
        prompt = f"""请为角色"{self.persona.name}"生成{num_contexts}个自然的对话场景上下文。

角色信息：
{self.persona}

要求：
1. 每个上下文应该是一个问题或情境描述
2. 涵盖日常对话、冲突、决策等不同场景
3. 足够具体，能引出角色的回应
4. 场景多样化

输出JSON格式：
{{"contexts": ["上下文1", "上下文2", ...]}}
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个对话场景设计专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("contexts", [])
    
    def _generate_pair_for_dimension(
        self, 
        context: str, 
        dimension: str
    ) -> Tuple[str, str]:
        """
        为特定维度生成正负例对
        
        Args:
            context: 对话上下文
            dimension: 要破坏的维度
            
        Returns:
            (正例, 负例) 元组
        """
        dim_descriptions = {
            "style": "语言风格",
            "value_system": "价值观和动机",
            "knowledge": "知识边界",
            "etiquette": "称呼礼仪"
        }
        
        tag = self.persona.get_tag(dimension)
        prototypes = self.persona.get_prototypes(dimension)
        
        prompt = f"""角色：{self.persona.name}
角色{dim_descriptions[dimension]}：{tag}

示例语句：
{chr(10).join(f'- {p}' for p in prototypes)}

对话场景：
{context}

请生成两个回复：
1. chosen（正例）：完全符合该角色{dim_descriptions[dimension]}的回复
2. rejected（负例）：在{dim_descriptions[dimension]}上有明显违背的回复（其他方面可以相似）

违背方式示例：
- style维度：使用完全不同的语气、句式、用词风格
- values维度：表达相反的价值观或不合理的动机
- knowledge维度：提到角色不该知道的知识，或对应该知道的事表现无知
- etiquette维度：使用错误的称呼或不符合关系的语气

要求：
1. 两个回复长度相似（1-3句话）
2. rejected仅在目标维度违背，不要全盘崩坏
3. 违背要明显但不夸张

输出JSON格式：
{{
  "chosen": "正例回复",
  "rejected": "负例回复",
  "violation_type": "具体违背了什么"
}}
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个角色对话生成专家，擅长创建对比样本。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result["chosen"], result["rejected"]
    
    def generate_stress_pairs(self, num_pairs: int = 10) -> List[PreferencePair]:
        """
        生成压力触发场景的偏好对
        
        Args:
            num_pairs: 生成数量
            
        Returns:
            偏好对列表
        """
        if not self.persona.stress_trigger:
            print("警告：该角色没有定义压力触发条件")
            return []
        
        pairs = []
        stress_contexts = self._generate_stress_contexts(num_pairs)
        
        for i, context in enumerate(stress_contexts):
            chosen, rejected = self._generate_stress_pair(context)
            
            pair = PreferencePair(
                context=context,
                chosen=chosen,
                rejected=rejected,
                violated_dimension="stress",
                metadata={"index": i, "dimension": "stress", "is_stress_scenario": True}
            )
            pairs.append(pair)
        
        return pairs
    
    def _generate_stress_contexts(self, num_contexts: int) -> List[str]:
        """生成压力场景上下文"""
        trigger = self.persona.stress_trigger
        
        prompt = f"""基于以下压力触发条件，生成{num_contexts}个具体的压力场景。

角色：{self.persona.name}
压力触发条件：{trigger.condition}
允许的变化：{trigger.allowed_change}

要求：
1. 每个场景都应该触发该压力条件
2. 场景具体且富有张力
3. 能引出角色在压力下的反应

输出JSON格式：
{{"contexts": ["场景1", "场景2", ...]}}
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个剧情设计专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("contexts", [])
    
    def _generate_stress_pair(self, context: str) -> Tuple[str, str]:
        """生成压力场景的正负例"""
        trigger = self.persona.stress_trigger
        
        prompt = f"""角色：{self.persona.name}
压力条件：{trigger.condition}
允许变化：{trigger.allowed_change}

场景：{context}

请生成两个回复：
1. chosen：符合允许变化范围的合理反应
2. rejected：超出允许变化范围的过激或不合理反应

输出JSON格式：
{{
  "chosen": "合理的压力反应",
  "rejected": "不合理的压力反应"
}}
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个角色心理分析专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result["chosen"], result["rejected"]
    
    def save_pairs_to_jsonl(self, pairs: List[PreferencePair], output_path: str):
        """保存偏好对到JSONL文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                data = {
                    "context": pair.context,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    "violated_dimension": pair.violated_dimension,
                    "metadata": pair.metadata
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"已保存 {len(pairs)} 个偏好对到 {output_path}")
    
    @staticmethod
    def load_pairs_from_jsonl(input_path: str) -> List[PreferencePair]:
        """从JSONL文件加载偏好对"""
        pairs = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                pair = PreferencePair(
                    context=data["context"],
                    chosen=data["chosen"],
                    rejected=data["rejected"],
                    violated_dimension=data["violated_dimension"],
                    metadata=data.get("metadata")
                )
                pairs.append(pair)
        return pairs


if __name__ == "__main__":
    # 测试示例
    from persona import Persona
    
    persona = Persona.from_json("../config/example_persona.json")
    generator = SampleGenerator(persona)
    
    # 生成10个偏好对
    pairs = generator.generate_preference_pairs(num_pairs=10)
    
    # 保存
    generator.save_pairs_to_jsonl(pairs, "../data/training_pairs.jsonl")
    
    # 打印第一个示例
    if pairs:
        print("\n示例偏好对：")
        print(f"上下文: {pairs[0].context}")
        print(f"正例: {pairs[0].chosen}")
        print(f"负例: {pairs[0].rejected}")
        print(f"违背维度: {pairs[0].violated_dimension}")