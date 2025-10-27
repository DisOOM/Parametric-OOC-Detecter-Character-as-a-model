"""
Persona 数据结构定义
用于存储和管理角色人格信息
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StressTrigger:
    """压力触发条件"""
    condition: str  # 触发条件描述
    allowed_change: str  # 允许的行为变化


@dataclass
class Persona:
    """角色人格定义"""
    name: str
    description: str = ""
    
    # 4个核心维度的标签
    tags: Dict[str, str] = field(default_factory=dict)
    # {
    #   "style": "风格描述",
    #   "values": "价值观描述",
    #   "knowledge": "知识边界描述",
    #   "etiquette": "称呼礼仪描述"
    # }
    
    # 压力触发条件
    stress_trigger: Optional[StressTrigger] = None
    
    # 原型片段 - 每个维度的示例语句
    prototypes: Dict[str, List[str]] = field(default_factory=dict)
    # {
    #   "style": ["示例1", "示例2", ...],
    #   "values": [...],
    #   "knowledge": [...],
    #   "etiquette": [...]
    # }
    
    @classmethod
    def from_json(cls, json_path: str) -> "Persona":
        """从JSON文件加载Persona"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 解析压力触发条件
        stress_trigger = None
        if "stress_trigger" in data:
            stress_trigger = StressTrigger(
                condition=data["stress_trigger"]["condition"],
                allowed_change=data["stress_trigger"]["allowed_change"]
            )
        
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            tags=data.get("tags", {}),
            stress_trigger=stress_trigger,
            prototypes=data.get("prototypes", {})
        )
    
    def to_json(self, json_path: str):
        """保存到JSON文件"""
        data = {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "prototypes": self.prototypes
        }
        
        if self.stress_trigger:
            data["stress_trigger"] = {
                "condition": self.stress_trigger.condition,
                "allowed_change": self.stress_trigger.allowed_change
            }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_dimension_names(self) -> List[str]:
        """获取所有维度名称"""
        return ["style", "values", "knowledge", "etiquette", "stress"]
    
    def get_tag(self, dimension: str) -> str:
        """获取指定维度的标签描述"""
        if dimension == "stress" and self.stress_trigger:
            return f"{self.stress_trigger.condition} -> {self.stress_trigger.allowed_change}"
        return self.tags.get(dimension, "")
    
    def get_prototypes(self, dimension: str) -> List[str]:
        """获取指定维度的原型片段"""
        return self.prototypes.get(dimension, [])
    
    def validate(self) -> bool:
        """验证Persona完整性"""
        required_dims = ["style", "values", "knowledge", "etiquette"]
        
        # 检查所有维度是否都有标签
        for dim in required_dims:
            if dim not in self.tags or not self.tags[dim]:
                print(f"警告: 缺少维度 '{dim}' 的标签")
                return False
        
        # 检查每个维度是否都有原型片段
        for dim in required_dims:
            if dim not in self.prototypes or len(self.prototypes[dim]) < 2:
                print(f"警告: 维度 '{dim}' 需要至少2条原型片段")
                return False
        
        return True
    
    def __str__(self) -> str:
        """字符串表示"""
        lines = [f"角色: {self.name}"]
        if self.description:
            lines.append(f"描述: {self.description}")
        
        lines.append("\n核心标签:")
        for dim, tag in self.tags.items():
            lines.append(f"  {dim}: {tag}")
        
        if self.stress_trigger:
            lines.append(f"\n压力触发: {self.stress_trigger.condition}")
            lines.append(f"  允许变化: {self.stress_trigger.allowed_change}")
        
        return "\n".join(lines)


def create_persona_from_dict(data: dict) -> Persona:
    """从字典创建Persona（用于程序化构建）"""
    stress_trigger = None
    if "stress_trigger" in data:
        stress_trigger = StressTrigger(
            condition=data["stress_trigger"]["condition"],
            allowed_change=data["stress_trigger"]["allowed_change"]
        )
    
    return Persona(
        name=data["name"],
        description=data.get("description", ""),
        tags=data.get("tags", {}),
        stress_trigger=stress_trigger,
        prototypes=data.get("prototypes", {})
    )