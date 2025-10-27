"""
配置加载器
从config.yaml或环境变量加载配置
"""

import os
import yaml
from typing import Optional, Dict, Any


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        # 默认配置
        default_config = {
            "openai": {
                "api_key": "",
                "model": "gpt-4",
                "base_url": ""
            },
            "training": {
                "model_name": "microsoft/deberta-v3-base",
                "output_dir": "./checkpoints",
                "num_epochs": 3,
                "batch_size": 8,
                "learning_rate": 2.0e-5,
                "max_length": 512
            },
            "sample_generation": {
                "num_pairs": 50,
                "num_contexts": 20,
                "temperature": 0.7
            },
            "scoring": {
                "threshold": 0.0,
                "threshold_per_dim": -0.5
            }
        }
        
        # 尝试加载YAML配置
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f) or {}
                    # 合并配置
                    self._deep_update(default_config, yaml_config)
            except Exception as e:
                print(f"警告: 无法加载配置文件 {self.config_path}: {e}")
                print("使用默认配置")
        
        return default_config
    
    def _deep_update(self, base: dict, update: dict):
        """深度更新字典"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def get_openai_api_key(self) -> Optional[str]:
        """获取OpenAI API密钥（优先环境变量）"""
        # 1. 环境变量（最高优先级）
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return env_key
        
        # 2. 配置文件
        config_key = self.config.get("openai", {}).get("api_key", "")
        if config_key:
            return config_key
        
        return None
    
    def get_openai_model(self) -> str:
        """获取OpenAI模型名称"""
        return self.config.get("openai", {}).get("model", "gpt-4")
    
    def get_openai_base_url(self) -> Optional[str]:
        """获取OpenAI API基础URL"""
        base_url = self.config.get("openai", {}).get("base_url", "")
        return base_url if base_url else None
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.config.get("training", {})
    
    def get_sample_generation_config(self) -> Dict[str, Any]:
        """获取样本生成配置"""
        return self.config.get("sample_generation", {})
    
    def get_scoring_config(self) -> Dict[str, Any]:
        """获取评分配置"""
        return self.config.get("scoring", {})
    
    def check_api_configured(self) -> bool:
        """检查API是否已配置"""
        api_key = self.get_openai_api_key()
        return api_key is not None and api_key != ""
    
    def print_config_guide(self):
        """打印配置指南"""
        if not self.check_api_configured():
            print("\n" + "=" * 70)
            print("⚠️  OpenAI API未配置")
            print("=" * 70)
            print("\n如需使用LLM功能（标签提取、样本生成），请配置API密钥：")
            print("\n方式1: 环境变量（推荐）")
            print("  export OPENAI_API_KEY='your-api-key-here'")
            print("\n方式2: 配置文件")
            print(f"  编辑 {self.config_path}")
            print("  填写 openai.api_key 字段")
            print("\n方式3: 直接传参")
            print("  TagExtractor(api_key='your-key')")
            print("  SampleGenerator(persona, api_key='your-key')")
            print("\n注意: quick_start.py 使用硬编码样本，不需要API")
            print("=" * 70 + "\n")


# 全局配置实例
_global_config = None

def get_config(config_path: str = "config/config.yaml") -> Config:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config


if __name__ == "__main__":
    # 测试配置加载
    config = get_config()
    
    print("配置加载测试：")
    print(f"  OpenAI API Key: {'已配置' if config.check_api_configured() else '未配置'}")
    print(f"  OpenAI Model: {config.get_openai_model()}")
    print(f"  训练配置: {config.get_training_config()}")
    
    config.print_config_guide()