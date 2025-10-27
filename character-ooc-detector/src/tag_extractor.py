"""
基于LLM的标签提取器
从自然语言人设描述中提取结构化标签
"""

import json
import os
from typing import Dict, List, Optional
from openai import OpenAI

from .config_loader import get_config


class TagExtractor:
    """使用LLM从文本中提取角色标签"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        初始化标签提取器
        
        Args:
            api_key: OpenAI API密钥（优先级：参数 > 环境变量 > config.yaml）
            model: 使用的模型名称（如果为None，从配置读取）
        """
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
        print(f"✓ TagExtractor initialized with model: {model}")
    
    def extract_tags_from_text(self, description: str) -> Dict:
        """
        从自然语言描述中提取标签
        
        Args:
            description: 角色的自然语言描述
            
        Returns:
            包含name, tags, stress_trigger, prototypes的字典
        """
        prompt = self._build_extraction_prompt(description)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个角色分析专家，擅长从人设描述中提取结构化信息。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    def _build_extraction_prompt(self, description: str) -> str:
        """构建提取提示词"""
        return f"""请分析以下角色描述，提取出结构化的标签信息。

角色描述：
{description}

请以JSON格式返回，包含以下字段：

1. name: 角色名称
2. description: 简短的角色概述（1-2句话）
3. tags: 包含4个维度的对象
   - style: 语言风格（词汇、句式、语气等）
   - values: 价值观和行为准则（信仰、底线、动机等）
   - knowledge: 知识边界（懂什么、不懂什么）
   - etiquette: 称呼礼仪（如何称呼不同关系的人）
4. stress_trigger: 压力触发条件（可选）
   - condition: 触发条件描述
   - allowed_change: 允许的行为变化
5. prototypes: 每个维度的示例语句（每个维度2-3条）
   - style: [示例1, 示例2, ...]
   - values: [...]
   - knowledge: [...]
   - etiquette: [...]

输出示例：
{{
  "name": "张三",
  "description": "江湖侠客，师从少林，重情重义",
  "tags": {{
    "style": "简洁直接，喜欢用短句",
    "values": "重视承诺，厌恶背叛",
    "knowledge": "熟悉武术，不懂现代科技",
    "etiquette": "称呼师父为'师傅'"
  }},
  "stress_trigger": {{
    "condition": "当师门遭遇背叛时",
    "allowed_change": "可以从冷静变得愤怒"
  }},
  "prototypes": {{
    "style": ["这事儿，我办。", "废话少说。"],
    "values": ["答应的事就得做到。"],
    "knowledge": ["这套拳法讲究以柔克刚。"],
    "etiquette": ["师傅教导的，不敢忘。"]
  }}
}}

请严格按照JSON格式输出，不要添加其他解释。"""
    
    def enrich_prototypes(self, persona_dict: Dict, num_per_dim: int = 3) -> Dict:
        """
        扩充原型片段数量
        
        Args:
            persona_dict: 现有的persona字典
            num_per_dim: 每个维度需要的片段数量
            
        Returns:
            扩充后的persona字典
        """
        prototypes = persona_dict.get("prototypes", {})
        tags = persona_dict.get("tags", {})
        
        for dim in ["style", "values", "knowledge", "etiquette"]:
            current = prototypes.get(dim, [])
            if len(current) < num_per_dim:
                # 生成更多片段
                new_examples = self._generate_more_examples(
                    dimension=dim,
                    tag_description=tags.get(dim, ""),
                    existing_examples=current,
                    num_needed=num_per_dim - len(current)
                )
                prototypes[dim] = current + new_examples
        
        persona_dict["prototypes"] = prototypes
        return persona_dict
    
    def _generate_more_examples(
        self, 
        dimension: str, 
        tag_description: str,
        existing_examples: List[str],
        num_needed: int
    ) -> List[str]:
        """生成更多示例片段"""
        dim_names = {
            "style": "语言风格",
            "values": "价值观",
            "knowledge": "知识范围",
            "etiquette": "称呼礼仪"
        }
        
        prompt = f"""请根据以下{dim_names[dimension]}描述，生成{num_needed}条符合该特征的示例语句。

{dim_names[dimension]}描述：{tag_description}

已有示例：
{chr(10).join(f'- {ex}' for ex in existing_examples)}

要求：
1. 新示例应该风格一致但内容不重复
2. 每条示例应该是完整的语句
3. 体现该维度的核心特征
4. 以JSON数组格式输出

输出格式：
{{"examples": ["示例1", "示例2", ...]}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个角色对话生成专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("examples", [])


def extract_tags_from_text(description: str, api_key: Optional[str] = None) -> Dict:
    """
    便捷函数：从文本提取标签
    
    Args:
        description: 角色描述文本
        api_key: OpenAI API密钥（可选）
        
    Returns:
        提取的标签字典
    """
    extractor = TagExtractor(api_key=api_key)
    return extractor.extract_tags_from_text(description)


if __name__ == "__main__":
    # 测试示例
    test_description = """
    李明是一个现代都市的程序员，25岁，性格内向但专业能力强。
    他说话比较直接，不太会拐弯抹角，经常使用技术术语。
    价值观上，他非常重视效率和逻辑，不喜欢无谓的社交。
    对技术领域非常熟悉，但对人际关系和情感表达比较生疏。
    称呼同事时比较随意，通常直呼其名或昵称。
    """
    
    extractor = TagExtractor()
    result = extractor.extract_tags_from_text(test_description)
    print(json.dumps(result, ensure_ascii=False, indent=2))