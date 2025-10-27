"""
Character OOC Detector - 角色人格一致性检测器

基于Reward Model的角色Out-of-Character检测系统
"""

__version__ = "0.1.0"

from .persona import Persona, StressTrigger, create_persona_from_dict
from .tag_extractor import TagExtractor, extract_tags_from_text
from .sample_generator import SampleGenerator, PreferencePair
from .model import MultiHeadRewardModel, RewardModelForTraining, create_reward_model, load_reward_model
from .trainer import RMTrainer
from .scorer import OOCScorer

__all__ = [
    # Persona相关
    "Persona",
    "StressTrigger",
    "create_persona_from_dict",
    
    # 标签提取
    "TagExtractor",
    "extract_tags_from_text",
    
    # 样本生成
    "SampleGenerator",
    "PreferencePair",
    
    # 模型
    "MultiHeadRewardModel",
    "RewardModelForTraining",
    "create_reward_model",
    "load_reward_model",
    
    # 训练
    "RMTrainer",
    
    # 评分
    "OOCScorer",
]