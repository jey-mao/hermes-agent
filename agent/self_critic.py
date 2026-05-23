"""
Hermes v0.4 — 认知层 SelfCritic
基于 Self-Improving-Agent 5维度评分 + Reflexion missing/superfluous。
"""
from dataclasses import dataclass, field


DIMENSIONS = ["准确性","完整性","效率","可操作性","新颖性"]
WEIGHTS = {"准确性":0.25,"完整性":0.20,"效率":0.10,"可操作性":0.30,"新颖性":0.15}


@dataclass
class Evaluation:
    score: float
    scores: dict
    weaknesses: list[str]
    classification: str
    missing: list[str] = field(default_factory=list)
    superfluous: list[str] = field(default_factory=list)
    retry_decision: str = "unknown"


@dataclass
class StepResult:
    step_id: str
    description: str
    tool: str
    action: str
    raw_output: str
    success: bool = True
    error_type: str = ""
    duration_ms: int = 0


class SelfCritic:
    def __init__(self, retry_threshold=5.0, replan_threshold=5.0, accept_threshold=8.0):
        self.retry_threshold = retry_threshold
        self.replan_threshold = replan_threshold
        self.accept_threshold = accept_threshold

    def evaluate(self, result: StepResult, expected: str = "") -> Evaluation:
        scores = {d: self._score(d, result, expected) for d in DIMENSIONS}
        weighted = sum(scores[d]*WEIGHTS[d] for d in DIMENSIONS)
        weaknesses = self._weaknesses(result, scores, expected)
        classification = self._classify(result)
        missing, superfluous = self._missing_superfluous(result, expected)
        decision = self._decide(weighted, classification, weaknesses)
        return Evaluation(
            score=round(weighted,2), scores={d:round(scores[d],1) for d in scores},
            weaknesses=weaknesses, classification=classification,
            missing=missing, superfluous=superfluous, retry_decision=decision)

    def _score(self, dim: str, result: StepResult, expected: str) -> float:
        out = result.raw_output.lower()
        err = result.error_type.lower()
        if dim == "准确性":
            if err: return 2.0
            return 3.0 if "error" in out or "失败" in out else 8.0
        if dim == "完整性":
            if not result.raw_output or len(result.raw_output) < 50: return 3.0
            # 有输出且长度够 → 至少6分（不强制要求完全匹配预期关键词）
            return 8.0
        if dim == "效率":
            if result.duration_ms > 30000: return 4.0
            if result.duration_ms > 10000: return 6.0
            return 8.0
        if dim == "可操作性":
            if err in ("network","timeout"): return 3.0
            if any(m in out for m in ["http","url","路径","完成","成功","data"]): return 8.0
            if "没有" in out or "0条" in out: return 4.0
            return 6.0
        return 6.0  # 新颖性默认

    def _weaknesses(self, result: StepResult, scores: dict, expected: str) -> list[str]:
        w = []
        for d, s in scores.items():
            if s < 6:
                if d == "准确性": w.append(f"执行出错({result.error_type})" if result.error_type else "输出含错误")
                elif d == "完整性": w.append("输出不完整")
                elif d == "效率": w.append(f"执行过慢({result.duration_ms//1000}秒)")
                elif d == "可操作性":
                    if "没有" in result.raw_output or "0条" in result.raw_output: w.append("搜索结果为空，需更换关键词")
                    else: w.append("输出无法直接使用")
        return w

    def _classify(self, result: StepResult) -> str:
        out = result.raw_output.lower()
        # quality: 工具成功但内容本身有问题（空/错误关键词）
        if any(k in out for k in ["没有", "no result", "0条", "找不到", "error", "失败"]):
            return "quality"
        # quality: 错误信息中包含"没有结果"
        if result.error_type and "no result" in result.error_type.lower():
            return "quality"
        if not result.success or result.error_type:
            err = result.error_type.lower() if result.error_type else ""
            if "timeout" in err: return "timeout"
            if any(k in err for k in ["connection","refused","network","dns"]): return "network"
            # tool: 工具本身不存在/不支持
            if any(k in err for k in ["not found", "not exist", "unsupported", "unknown tool"]):
                return "tool"
            if any(k in err for k in ["auth","permission","403","401"]): return "auth"
            return "unknown"
        return "unknown"

    def _missing_superfluous(self, result: StepResult, expected: str) -> tuple:
        missing, superfluous = [], []
        if expected and len(result.raw_output) < 100 and expected[:10] not in result.raw_output:
            missing.append(f"未找到：{expected[:50]}")
        if len(result.raw_output) > 50000: superfluous.append("输出过长")
        if "error" in result.raw_output.lower() or "traceback" in result.raw_output.lower():
            missing.append("执行出错，无有效输出")
        return missing, superfluous

    def _decide(self, weighted: float, classification: str, weaknesses: list) -> str:
        if weighted >= self.accept_threshold: return "accept"
        if classification in ("network","timeout") and weighted >= self.retry_threshold: return "retry"
        if weighted < self.replan_threshold: return "replan"
        if classification in ("auth","unknown"): return "human_review"
        if weaknesses and len(weaknesses) <= 2: return "retry"
        return "replan"
