"""
Reward Model训练器
支持偏好对训练和多维度评分
"""

import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
from typing import List, Dict, Optional

from .model import MultiHeadRewardModel, RewardModelForTraining, create_reward_model
from .sample_generator import PreferencePair


class PreferencePairDataset(Dataset):
    """偏好对数据集"""
    
    def __init__(
        self,
        pairs: List[PreferencePair],
        tokenizer,
        max_length: int = 512
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 维度名到索引的映射（移除无效的stress维度）
        self.dim_to_idx = {
            "style": 0,
            "value_system": 1,
            "knowledge": 2,
            "etiquette": 3
        }
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # 构建输入文本：上下文 + 回复
        chosen_text = f"{pair.context}\n回复: {pair.chosen}"
        rejected_text = f"{pair.context}\n回复: {pair.rejected}"
        
        # Tokenize
        chosen_encodings = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_encodings = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 违背的维度
        violated_dim_idx = self.dim_to_idx.get(pair.violated_dimension, 0)
        
        return {
            "chosen_input_ids": chosen_encodings["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encodings["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encodings["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encodings["attention_mask"].squeeze(0),
            "violated_dimension": torch.tensor(violated_dim_idx, dtype=torch.long)
        }


class RMTrainer:
    """Reward Model训练器"""
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        output_dir: str = "./checkpoints",
        device: str = None,
        pretrained_rm_path: Optional[str] = None
    ):
        """
        初始化训练器
        
        Args:
            model_name: 预训练模型名称
            output_dir: 输出目录
            device: 设备（cuda/cpu）
            pretrained_rm_path: 预训练RM路径（可选，用于微调）
        """
        self.model_name = model_name
        self.pretrained_rm_path = pretrained_rm_path
        self.output_dir = output_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"使用设备: {self.device}")
    
    def train(
        self,
        train_pairs: List[PreferencePair],
        val_pairs: Optional[List[PreferencePair]] = None,
        num_epochs: int = 10,
        batch_size: int = 6,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        max_length: int = 512,
        save_steps: int = 500,
        eval_steps: int = 500,
        use_bf16: bool = True,
        seed: Optional[int] = 42,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 3
    ):
        """
        训练模型
        
        Args:
            train_pairs: 训练偏好对
            val_pairs: 验证偏好对（可选）
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            warmup_steps: warmup步数
            max_length: 最大序列长度
            save_steps: 保存间隔
            eval_steps: 评估间隔
            use_bf16: 是否使用bf16混合精度（默认True，节省显存）
            seed: 随机种子（默认42，设为None禁用）
            weight_decay: L2正则化系数（默认0.01，防止过拟合）
            early_stopping_patience: 早停patience（验证loss不下降的epoch数）
        """
        # 设置随机种子以保证可复现性
        if seed is not None:
            print(f"\n🎲 设置随机种子: {seed}")
            set_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # 设置确定性算法（可能略微影响性能）
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"\n开始训练...")
        print(f"训练样本数: {len(train_pairs)}")
        if val_pairs:
            print(f"验证样本数: {len(val_pairs)}")
        
        # 创建数据集
        train_dataset = PreferencePairDataset(train_pairs, self.tokenizer, max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = None
        if val_pairs:
            val_dataset = PreferencePairDataset(val_pairs, self.tokenizer, max_length)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # 创建模型
        base_model = create_reward_model(
            model_name=self.model_name,
            pretrained_rm_path=self.pretrained_rm_path
        )
        model = RewardModelForTraining(base_model)
        
        # 混合精度设置
        if use_bf16 and self.device == "cuda":
            if torch.cuda.is_bf16_supported():
                print("✓ 使用BF16混合精度训练（节省显存）")
                model = model.to(self.device, dtype=torch.bfloat16)
                use_amp = True
                amp_dtype = torch.bfloat16
            else:
                print("⚠️  当前GPU不支持BF16，使用FP32")
                model = model.to(self.device)
                use_amp = False
                amp_dtype = torch.float32
        else:
            model = model.to(self.device)
            use_amp = False
            amp_dtype = torch.float32
        
        # 显示模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 估算显存占用（粗略）
        bytes_per_param = 2 if use_amp else 4  # bf16: 2 bytes, fp32: 4 bytes
        model_size_mb = (total_params * bytes_per_param) / (1024**2)
        
        print(f"\n📊 模型参数统计:")
        print(f"  基础模型: {self.model_name}")
        print(f"  总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  可训练参数: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"  精度: {'BF16' if use_amp else 'FP32'}")
        print(f"  预估显存: ~{model_size_mb:.0f}MB (仅模型权重)")
        print(f"  设备: {self.device}")
        
        # 优化器和调度器（添加weight decay正则化）
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        print(f"\n⚙️  训练配置:")
        print(f"  学习率: {learning_rate}")
        print(f"  Weight decay (L2): {weight_decay}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练循环（添加early stopping）
        global_step = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            # 训练
            model.train()
            train_loss = 0
            train_metrics = {
                "overall_loss": 0,
                "style_loss": 0,
                "value_system_loss": 0,
                "knowledge_loss": 0,
                "etiquette_loss": 0
            }
            
            progress_bar = tqdm(train_loader, desc="训练中")
            for batch in progress_bar:
                # 移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 混合精度训练
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        outputs = model(**batch)
                        loss = outputs["loss"]
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                else:
                    # 正常精度训练
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                
                # 记录指标
                train_loss += loss.item()
                for key in train_metrics.keys():
                    if key in outputs:
                        train_metrics[key] += outputs[key].item()
                
                global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # 定期评估
                if val_loader and global_step % eval_steps == 0:
                    val_loss, val_metrics = self._evaluate(model, val_loader)
                    print(f"\n验证 @ step {global_step}:")
                    print(f"  验证loss: {val_loss:.4f}")
                    
                    # 保存最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self._save_model(model.model, "best_model")
                        print(f"  保存最佳模型 (loss: {val_loss:.4f})")
                    
                    model.train()
                
                # 定期保存
                if global_step % save_steps == 0:
                    self._save_model(model.model, f"checkpoint-{global_step}")
            
            # Epoch结束统计
            avg_train_loss = train_loss / len(train_loader)
            print(f"\n训练loss: {avg_train_loss:.4f}")
            
            for key in train_metrics:
                avg_metric = train_metrics[key] / len(train_loader)
                print(f"  {key}: {avg_metric:.4f}")
            
            # Epoch结束时评估（添加early stopping）
            if val_loader:
                val_loss, val_metrics = self._evaluate(model, val_loader)
                print(f"\nEpoch {epoch + 1} 验证结果:")
                print(f"  总体loss: {val_loss:.4f}")
                for key, value in val_metrics.items():
                    print(f"  {key}: {value:.4f}")
                
                # Early stopping检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"  ✓ 新的最佳验证loss: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"  ⚠️  验证loss未改善 ({patience_counter}/{early_stopping_patience})")
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"\n⏹  Early stopping triggered after {epoch + 1} epochs")
                        print(f"  最佳验证loss: {best_val_loss:.4f}")
                        break
        
        # 保存最终模型
        self._save_model(model.model, "final_model")
        print(f"\n训练完成！模型已保存到 {self.output_dir}")
    
    def _evaluate(self, model, val_loader):
        """评估模型"""
        model.eval()
        total_loss = 0
        metrics = {
            "overall_loss": 0,
            "style_loss": 0,
            "value_system_loss": 0,
            "knowledge_loss": 0,
            "etiquette_loss": 0,
            "stress_loss": 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                
                total_loss += outputs["loss"].item()
                for key in metrics.keys():
                    if key in outputs:
                        metrics[key] += outputs[key].item()
        
        avg_loss = total_loss / len(val_loader)
        for key in metrics:
            metrics[key] /= len(val_loader)
        
        return avg_loss, metrics
    
    def _save_model(self, model: MultiHeadRewardModel, name: str):
        """保存模型"""
        save_dir = os.path.join(self.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型权重
        torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        
        # 保存配置
        model.config.save_pretrained(save_dir)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        # 保存训练信息
        info = {
            "model_name": self.model_name,
            "dimension_names": model.dimension_names,
            "num_dimensions": model.num_dimensions
        }
        with open(os.path.join(save_dir, "training_info.json"), 'w') as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    # 测试训练器
    from sample_generator import PreferencePair
    
    # 创建虚拟数据
    print("创建测试数据...")
    dummy_pairs = [
        PreferencePair(
            context="你最近怎么样？",
            chosen="还行，在练新招式。",
            rejected="超级棒！我最近学了好多现代科技知识呢！",
            violated_dimension="knowledge"
        )
        for _ in range(50)
    ]
    
    # 创建训练器
    trainer = RMTrainer(
        model_name="microsoft/deberta-v3-base",
        output_dir="./test_checkpoints"
        # pretrained_rm_path="OpenAssistant/reward-model-deberta-v3-base" # 可选测试
    )
    
    # 训练
    print("\n开始测试训练...")
    trainer.train(
        train_pairs=dummy_pairs[:40],
        val_pairs=dummy_pairs[40:],
        num_epochs=1,
        batch_size=4,
        save_steps=20,
        eval_steps=20
    )
    
    print("\n训练器测试完成！")