"""
OOC评分器
使用训练好的RM模型对文本进行一致性评分
"""

import torch
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import numpy as np

from .model import MultiHeadRewardModel, load_reward_model
from .persona import Persona


class OOCScorer:
    """Out-of-Character评分器"""
    
    def __init__(
        self,
        model_path: str,
        device: str = None,
        threshold: float = 0.0
    ):
        """
        初始化评分器
        
        Args:
            model_path: 训练好的模型路径
            device: 计算设备
            threshold: OOC判定阈值（分数低于此值视为OOC）
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        
        # 加载模型和tokenizer
        print(f"从 {model_path} 加载模型...")
        self.model = load_reward_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 维度名称（移除无效的stress维度）
        self.dimension_names = ["style", "value_system", "knowledge", "etiquette"]
        
        print(f"模型加载完成，使用设备: {self.device}")
    
    def score(
        self,
        context: str,
        response: str,
        persona: Optional[Persona] = None,
        return_details: bool = True
    ) -> Dict:
        """
        对单个回复进行评分
        
        Args:
            context: 对话上下文
            response: 角色的回复
            persona: 角色人格（可选，用于显示）
            return_details: 是否返回详细信息
            
        Returns:
            包含评分结果的字典
        """
        # 构建输入文本
        text = f"{context}\n回复: {response}"
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 移动到设备
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # 提取分数
        overall_score = outputs["overall_score"].cpu().item()
        dimension_scores = {
            dim: outputs["dimension_scores"][dim].cpu().item()
            for dim in self.dimension_names
        }
        
        # 判断是否通过
        passed = overall_score >= self.threshold
        
        # 找出最低分的维度
        min_dim = min(dimension_scores.items(), key=lambda x: x[1])
        
        result = {
            "overall_score": overall_score,
            "dimension_scores": dimension_scores,
            "passed": passed,
            "weakest_dimension": min_dim[0],
            "weakest_score": min_dim[1]
        }
        
        if return_details:
            result["context"] = context
            result["response"] = response
            result["threshold"] = self.threshold
            
            if persona:
                result["character_name"] = persona.name
        
        return result
    
    def score_batch(
        self,
        contexts: List[str],
        responses: List[str],
        batch_size: int = 8
    ) -> List[Dict]:
        """
        批量评分
        
        Args:
            contexts: 上下文列表
            responses: 回复列表
            batch_size: 批次大小
            
        Returns:
            评分结果列表
        """
        results = []
        
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]
            
            # 构建批次文本
            texts = [
                f"{ctx}\n回复: {resp}"
                for ctx, resp in zip(batch_contexts, batch_responses)
            ]
            
            # Tokenize
            encodings = self.tokenizer(
                texts,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # 处理每个样本
            overall_scores = outputs["overall_score"].cpu().numpy()
            
            for j in range(len(batch_contexts)):
                dimension_scores = {
                    dim: outputs["dimension_scores"][dim][j].cpu().item()
                    for dim in self.dimension_names
                }
                
                overall_score = overall_scores[j]
                min_dim = min(dimension_scores.items(), key=lambda x: x[1])
                
                result = {
                    "overall_score": float(overall_score),
                    "dimension_scores": dimension_scores,
                    "passed": bool(overall_score >= self.threshold),
                    "weakest_dimension": min_dim[0],
                    "weakest_score": min_dim[1],
                    "context": batch_contexts[j],
                    "response": batch_responses[j]
                }
                results.append(result)
        
        return results
    
    def compare_responses(
        self,
        context: str,
        responses: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict:
        """
        比较多个候选回复
        
        Args:
            context: 对话上下文
            responses: 候选回复列表
            labels: 回复标签（可选）
            
        Returns:
            比较结果
        """
        scores = []
        
        for response in responses:
            result = self.score(context, response, return_details=False)
            scores.append(result)
        
        # 排序
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i]["overall_score"],
            reverse=True
        )
        
        return {
            "context": context,
            "responses": responses,
            "labels": labels,
            "scores": scores,
            "ranked_indices": ranked_indices,
            "best_response": responses[ranked_indices[0]],
            "best_score": scores[ranked_indices[0]]["overall_score"]
        }
    
    def analyze_violations(
        self,
        context: str,
        response: str,
        persona: Persona,
        threshold_per_dim: float = -0.5
    ) -> Dict:
        """
        分析具体的OOC违背情况
        
        Args:
            context: 上下文
            response: 回复
            persona: 角色人格
            threshold_per_dim: 每个维度的阈值
            
        Returns:
            详细的违背分析
        """
        result = self.score(context, response, persona)
        
        violations = []
        
        for dim in self.dimension_names:
            score = result["dimension_scores"][dim]
            if score < threshold_per_dim:
                violation = {
                    "dimension": dim,
                    "score": score,
                    "expected": persona.get_tag(dim),
                    "examples": persona.get_prototypes(dim)
                }
                violations.append(violation)
        
        analysis = {
            **result,
            "violations": violations,
            "num_violations": len(violations),
            "is_ooc": len(violations) > 0
        }
        
        return analysis
    
    def get_dimension_importance(self) -> Dict[str, float]:
        """
        获取各维度的重要性权重（基于模型参数）
        
        Returns:
            维度重要性字典
        """
        # 简单实现：基于各head的参数量
        importance = {}
        
        for dim in self.dimension_names:
            head = self.model.score_heads[dim]
            num_params = sum(p.numel() for p in head.parameters())
            importance[dim] = num_params
        
        # 归一化
        total = sum(importance.values())
        importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def format_result(self, result: Dict, verbose: bool = True) -> str:
        """
        格式化输出结果
        
        Args:
            result: score()返回的结果
            verbose: 是否详细输出
            
        Returns:
            格式化的字符串
        """
        lines = []
        
        if "character_name" in result:
            lines.append(f"角色: {result['character_name']}")
        
        if verbose and "context" in result:
            lines.append(f"\n上下文: {result['context']}")
            lines.append(f"回复: {result['response']}")
        
        lines.append(f"\n总体评分: {result['overall_score']:.3f}")
        lines.append(f"是否通过: {'✓ 是' if result['passed'] else '✗ 否'}")
        
        lines.append(f"\n各维度评分:")
        for dim, score in result['dimension_scores'].items():
            status = "✓" if score >= 0 else "✗"
            lines.append(f"  {dim:12s}: {score:6.3f} {status}")
        
        lines.append(f"\n最弱维度: {result['weakest_dimension']} ({result['weakest_score']:.3f})")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试评分器（需要先训练模型）
    import os
    
    # 检查是否存在训练好的模型
    if os.path.exists("./test_checkpoints/best_model"):
        print("加载测试模型...")
        scorer = OOCScorer(
            model_path="./test_checkpoints/best_model",
            threshold=0.0
        )
        
        # 测试评分
        context = "你最近怎么样？"
        response_good = "还行，在练新招式。师傅说我进步挺快。"
        response_bad = "超级棒！我在用ChatGPT学习编程呢！"
        
        print("\n=== 测试正例 ===")
        result1 = scorer.score(context, response_good)
        print(scorer.format_result(result1))
        
        print("\n=== 测试负例 ===")
        result2 = scorer.score(context, response_bad)
        print(scorer.format_result(result2))
        
        print("\n=== 比较多个回复 ===")
        comparison = scorer.compare_responses(
            context,
            [response_good, response_bad],
            ["符合人设", "违背人设"]
        )
        print(f"最佳回复: {comparison['best_response']}")
        print(f"最佳分数: {comparison['best_score']:.3f}")
    else:
        print("未找到训练好的模型，请先运行trainer.py训练模型")