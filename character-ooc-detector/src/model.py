"""
多头Reward Model架构
包含5个评分头：style, values, knowledge, etiquette, stress
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel, AutoModelForSequenceClassification
from typing import Dict, Optional


class MultiHeadRewardModel(PreTrainedModel):
    """
    多头奖励模型
    
    架构：
    - 共享编码器（DeBERTa-v3-base）
    - 4个独立的评分头（每个维度一个）
    - 每个头输出一个标量分数
    """
    
    def __init__(self, config, num_dimensions: int = 4, dropout_rate: float = 0.1):
        """
        初始化模型
        
        Args:
            config: transformer配置
            num_dimensions: 维度数量（默认4：style, value_system, knowledge, etiquette）
            dropout_rate: Dropout比率（防止过拟合）
        """
        super().__init__(config)
        
        self.num_dimensions = num_dimensions
        self.dimension_names = ["style", "value_system", "knowledge", "etiquette"]
        self.dropout_rate = dropout_rate
        
        # 共享编码器
        self.encoder = AutoModel.from_config(config)
        
        # 每个维度一个评分头（增强正则化）
        # 每个维度一个评分头（简化结构以匹配预训练RM）
        self.score_heads = nn.ModuleDict({
            dim: nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(config.hidden_size, 1)
            )
            for dim in self.dimension_names
        })
        
        # 总体评分头（简化结构）
        self.overall_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(config.hidden_size, 1)
        )
        
        # 初始化权重
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            return_dict: 是否返回字典
            
        Returns:
            包含各维度分数的字典
        """
        # 编码
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 使用[CLS] token的表示
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # 计算每个维度的分数
        dimension_scores = {}
        for dim_name in self.dimension_names:
            score = self.score_heads[dim_name](pooled_output).squeeze(-1)  # [batch_size]
            dimension_scores[dim_name] = score
        
        # 计算总体分数
        overall_score = self.overall_head(pooled_output).squeeze(-1)  # [batch_size]
        
        if return_dict:
            return {
                "overall_score": overall_score,
                "dimension_scores": dimension_scores,
                "pooled_output": pooled_output
            }
        else:
            return overall_score, dimension_scores


class RewardModelForTraining(nn.Module):
    """
    训练用的Reward Model包装器
    支持偏好对比学习
    """
    
    def __init__(self, base_model: MultiHeadRewardModel):
        super().__init__()
        self.model = base_model
    
    def forward(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        violated_dimension: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        计算偏好对的loss
        
        Args:
            chosen_input_ids: 正例输入
            chosen_attention_mask: 正例mask
            rejected_input_ids: 负例输入
            rejected_attention_mask: 负例mask
            violated_dimension: 违背的维度索引 [batch_size]
            
        Returns:
            包含loss和分数的字典
        """
        # 前向传播正例
        chosen_outputs = self.model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask
        )
        
        # 前向传播负例
        rejected_outputs = self.model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        )
        
        # 计算ranking loss (chosen应该比rejected分数高)
        losses = {}
        
        # 1. 总体ranking loss
        overall_loss = -torch.log(
            torch.sigmoid(chosen_outputs["overall_score"] - rejected_outputs["overall_score"])
        ).mean()
        losses["overall_loss"] = overall_loss
        
        # 2. 各维度的ranking loss
        dimension_losses = []
        for dim_name in self.model.dimension_names:
            chosen_score = chosen_outputs["dimension_scores"][dim_name]
            rejected_score = rejected_outputs["dimension_scores"][dim_name]
            
            dim_loss = -torch.log(
                torch.sigmoid(chosen_score - rejected_score)
            ).mean()
            
            losses[f"{dim_name}_loss"] = dim_loss
            dimension_losses.append(dim_loss)
        
        # 3. 维度加权loss（如果提供了violated_dimension）
        if violated_dimension is not None:
            # 对违背的维度给予更高权重
            weighted_dim_loss = 0
            for i, dim_name in enumerate(self.model.dimension_names):
                weight = torch.where(violated_dimension == i, 2.0, 1.0).float()
                dim_loss = -torch.log(
                    torch.sigmoid(
                        chosen_outputs["dimension_scores"][dim_name] - 
                        rejected_outputs["dimension_scores"][dim_name]
                    )
                )
                weighted_dim_loss += (dim_loss * weight).mean()
            
            losses["weighted_dimension_loss"] = weighted_dim_loss / len(self.model.dimension_names)
        
        # 总loss
        total_loss = overall_loss + sum(dimension_losses) / len(dimension_losses)
        if violated_dimension is not None:
            total_loss = 0.5 * total_loss + 0.5 * losses["weighted_dimension_loss"]
        
        losses["loss"] = total_loss
        
        return {
            **losses,
            "chosen_scores": chosen_outputs,
            "rejected_scores": rejected_outputs
        }


def create_reward_model(
    model_name: str = "microsoft/deberta-v3-base",
    pretrained_rm_path: Optional[str] = None
) -> MultiHeadRewardModel:
    """
    创建多头Reward Model
    
    Args:
        model_name: 预训练模型名称
        pretrained_rm_path: 预训练RM的路径 (可选, 用于微调)
        
    Returns:
        初始化的模型
    """
    config = AutoConfig.from_pretrained(pretrained_rm_path or model_name)
    model = MultiHeadRewardModel(config)
    
    if pretrained_rm_path:
        print(f"🔄 从预训练奖励模型 '{pretrained_rm_path}' 加载权重进行微调...")
        
        # 加载完整的预训练RM
        try:
            pretrained_rm = AutoModelForSequenceClassification.from_pretrained(pretrained_rm_path)
        except Exception as e:
            print(f"❌ 加载预训练RM '{pretrained_rm_path}' 失败: {e}")
            print("    将回退到从基础模型初始化。")
            pretrained = AutoModel.from_pretrained(model_name)
            model.encoder = pretrained
            return model

        # 1. 复制encoder权重
        #    不同模型的base model属性名不同 (e.g., 'deberta', 'roberta', 'base_model')
        encoder_loaded = False
        for attr in ['deberta', 'roberta', 'base_model']:
            if hasattr(pretrained_rm, attr):
                model.encoder = getattr(pretrained_rm, attr)
                encoder_loaded = True
                print(f"    - ✓ 成功复制 '{attr}' 的编码器权重。")
                break
        if not encoder_loaded:
            print("    - ⚠️  警告: 无法自动确定encoder，跳过encoder权重加载。")

        # 2. 将预训练RM的分类头权重复制到我们所有的头上
        if hasattr(pretrained_rm, "classifier") and isinstance(pretrained_rm.classifier, nn.Linear):
            rm_head_state_dict = pretrained_rm.classifier.state_dict()
            
            for head in model.score_heads.values():
                # head[1] is nn.Linear
                head[1].load_state_dict(rm_head_state_dict)
            
            model.overall_head[1].load_state_dict(rm_head_state_dict)
            print("    - ✓ 成功将预训练RM的评分头权重复制到所有头。")
        else:
            print("    - ⚠️  警告: 未找到兼容的评分头 (nn.Linear)，评分头将随机初始化。")
        
        print("    - 权重加载完成。")

    else:
        print(f"🔄 从基础模型 '{model_name}' 初始化权重...")
        # 加载预训练权重到encoder
        pretrained = AutoModel.from_pretrained(model_name)
        model.encoder = pretrained
    
    return model


def load_reward_model(checkpoint_path: str) -> MultiHeadRewardModel:
    """
    从checkpoint加载模型
    
    Args:
        checkpoint_path: checkpoint路径
        
    Returns:
        加载的模型
    """
    # 首先加载配置
    config = AutoConfig.from_pretrained(checkpoint_path)
    
    # 动态确定维度数量
    training_info_path = os.path.join(checkpoint_path, "training_info.json")
    if os.path.exists(training_info_path):
        with open(training_info_path, 'r') as f:
            info = json.load(f)
            num_dimensions = info.get("num_dimensions", 4) # 默认为4
    else:
        num_dimensions = 4
        
    # 创建模型并加载权重
    model = MultiHeadRewardModel(config, num_dimensions=num_dimensions)
    
    try:
        state_dict = torch.load(
            os.path.join(checkpoint_path, "pytorch_model.bin"),
            map_location="cpu"
        )
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"⚠️  加载模型权重时出现不匹配: {e}")
        print("    尝试以非严格模式加载...")
        model.load_state_dict(state_dict, strict=False)

    return model


if __name__ == "__main__":
    # 测试模型
    print("创建模型...")
    model = create_reward_model()
    
    # 测试前向传播
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print("\n测试前向传播...")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"Overall score shape: {outputs['overall_score'].shape}")
    print(f"Dimension scores:")
    for dim, score in outputs['dimension_scores'].items():
        print(f"  {dim}: {score.shape}")
    
    # 测试训练包装器
    print("\n测试训练模式...")
    training_model = RewardModelForTraining(model)
    
    chosen_ids = torch.randint(0, 30000, (batch_size, seq_len))
    rejected_ids = torch.randint(0, 30000, (batch_size, seq_len))
    chosen_mask = torch.ones(batch_size, seq_len)
    rejected_mask = torch.ones(batch_size, seq_len)
    violated_dim = torch.tensor([0, 1])  # style, values
    
    train_outputs = training_model(
        chosen_input_ids=chosen_ids,
        chosen_attention_mask=chosen_mask,
        rejected_input_ids=rejected_ids,
        rejected_attention_mask=rejected_mask,
        violated_dimension=violated_dim
    )
    
    print(f"Total loss: {train_outputs['loss'].item():.4f}")
    print(f"Overall loss: {train_outputs['overall_loss'].item():.4f}")
    for dim in model.dimension_names:
        print(f"{dim} loss: {train_outputs[f'{dim}_loss'].item():.4f}")
    
    print("\n模型测试完成！")