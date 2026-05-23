"""
Hermes v0.4 — 规划层 TaskPlanner
基于 AIOps-Pilot planner 函数启发，支持分级分解和重规划。
"""
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TaskSignature:
    domain: str
    action: str
    constraints: str
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            raw = f"{self.domain}|{self.action}|{self.constraints}"
            self.hash = hashlib.md5(raw.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        return {"domain": self.domain, "action": self.action,
                "constraints": self.constraints, "hash": self.hash}

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSignature":
        return cls(domain=d["domain"], action=d["action"],
                    constraints=d["constraints"], hash=d.get("hash", ""))


@dataclass
class Step:
    step_id: str
    description: str
    tool: str
    action: str
    expected: str
    status: Literal["pending","running","done","failed"] = "pending"
    result: str = ""
    score: float = -1.0


@dataclass
class ExecutionPlan:
    plan_id: str
    task_sig: TaskSignature
    steps: list[Step]
    current_step: int = 0
    replan_count: int = 0
    status: Literal["planning","executing","criticizing","complete","failed","replanning"] = "planning"
    history: list[dict] = field(default_factory=list)

    def next_step(self) -> Step | None:
        if self.current_step >= len(self.steps):
            return None
        step = self.steps[self.current_step]
        step.status = "running"
        return step

    def advance(self):
        if self.current_step < len(self.steps):
            self.steps[self.current_step].status = "done"
        self.current_step += 1
        if self.current_step >= len(self.steps):
            self.status = "complete"


class TaskPlanner:
    """规划层：任务分解 + 重规划 + 历史复用"""

    def __init__(self, llm_client=None, evolution_engine=None):
        self.llm = llm_client
        self.ev_engine = evolution_engine

    def plan(self, user_message: str) -> ExecutionPlan:
        task_sig = self._extract_signature(user_message)

        best_steps = []
        best_score = None
        if self.ev_engine:
            best = self.ev_engine.find_similar(task_sig)
            if best:
                best_steps = best.get("steps", [])
                best_score = best.get("score")

        steps = self._generate_steps(user_message, best_steps, best_score)
        return ExecutionPlan(
            plan_id=str(uuid.uuid4()),
            task_sig=task_sig,
            steps=steps,
        )

    def _extract_signature(self, message: str) -> TaskSignature:
        m = message.lower()
        if any(k in m for k in ["1688","货源","阿里巴巴"]): domain = "1688_货源采集"
        elif any(k in m for k in ["tiktok","tk","东南亚","shopee","lazada"]): domain = "TikTok_电商选品"
        elif any(k in m for k in ["github","git","commit","pr"]): domain = "GitHub_开发"
        elif any(k in m for k in ["搜索","google","browser","网页"]): domain = "browser_信息搜索"
        elif any(k in m for k in ["飞书","feishu","消息"]): domain = "飞书_协作"
        elif any(k in m for k in ["架构","框架","framework"]): domain = "架构_设计"
        else: domain = "通用_任务"

        if any(k in m for k in ["采集","抓取","爬"]): action = "采集"
        elif any(k in m for k in ["搜索","找","查"]): action = "搜索"
        elif any(k in m for k in ["对比","分析","评估"]): action = "分析"
        elif any(k in m for k in ["写","生成","创建"]): action = "生成"
        elif any(k in m for k in ["修复","改","解决"]): action = "修复"
        else: action = "执行"

        constraints = self._extract_constraints(message)
        return TaskSignature(domain=domain, action=action, constraints=constraints)

    def _extract_constraints(self, message: str) -> str:
        parts = []
        for name, code in {"马来西亚":"MY","菲律宾":"PH","泰国":"TH","越南":"VN","新加坡":"SG","印尼":"ID"}.items():
            if name in message: parts.append(f"国家={name}({code})")
        for p in re.findall(r"RM?-?\s*(\d+[-\d]*)", message): parts.append(f"价格={p}")
        for s in re.findall(r"月销\s*(\d+)", message): parts.append(f"月销≥{s}")
        for c in re.findall(r"(蓝牙耳机|充电宝|手机壳|数据线|手机支架|3C|数码|家居|美妆|服装)", message): parts.append(f"品类={c}")
        return "；".join(parts) if parts else "无特定约束"

    def _generate_steps(self, message: str, best_steps: list, best_score: float) -> list[Step]:
        if best_steps and best_score is not None and best_score >= 7.0:
            prompt = f"任务：{message}\n历史最佳（评分{best_score}）：{json.dumps(best_steps[:5],ensure_ascii=False)}\n请微调后输出JSON数组，每步含step_id/description/tool/action/expected。"
        else:
            prompt = f"任务：{message}\n请分解为3-7个可执行步骤，每步含step_id/description/tool/action/expected。输出JSON数组。"
        raw = self.llm.generate(prompt, schema="json") if self.llm else ""
        return self._parse_steps(raw)

    def _parse_steps(self, raw: str) -> list[Step]:
        try:
            for mark in ["```json","```"]:
                if mark in raw:
                    raw = raw.split(mark)[1].split("```")[0].strip()
                    break
            data = json.loads(raw)
            if isinstance(data, dict) and "steps" in data: data = data["steps"]
            return [Step(step_id=i.get("step_id", str(uuid.uuid4())[:8]),
                         description=i.get("description",""),
                         tool=i.get("tool","terminal"),
                         action=i.get("action",""),
                         expected=i.get("expected","")) for i in data] or self._fallback()
        except Exception:
            return self._fallback()

    def _fallback(self) -> list[Step]:
        return [Step(step_id=str(uuid.uuid4())[:8], description="执行任务",
                     tool="terminal", action="执行", expected="完成")]

    def replan(self, plan: ExecutionPlan, failed_step: Step, reason: str, new_info: str = "") -> ExecutionPlan:
        plan.history.append({"failed_step_id": failed_step.step_id, "reason": reason, "new_info": new_info})
        plan.replan_count += 1
        cls_ = self._classify(reason)
        if cls_ == "single":
            idx = next((i for i,s in enumerate(plan.steps) if s.step_id==failed_step.step_id), -1)
            if idx >= 0:
                plan.steps[idx] = self._replace(failed_step, reason)
                plan.steps[idx].status = "pending"
                plan.current_step = idx
                plan.status = "executing"
        elif cls_ == "method":
            new_s = self._generate_steps(f"失败({reason})换方法：{failed_step.description}", [], None)
            plan.steps = plan.steps[:plan.current_step] + new_s + plan.steps[plan.current_step+1:]
            plan.status = "executing"
        elif cls_ == "newinfo":
            from dataclasses import replace
            plan.steps.insert(plan.current_step+1, Step(
                step_id=str(uuid.uuid4())[:8], description=f"整合新信息：{new_info[:80]}",
                tool="terminal", action="整合", expected="信息已整合"))
            plan.status = "executing"
        else:
            if plan.replan_count <= 1:
                plan.steps = self._generate_steps(f"全面重建，原因：{reason}", [], None)
                plan.current_step = 0
                plan.status = "executing"
            else:
                plan.status = "failed"
        return plan

    def _classify(self, reason: str) -> str:
        r = reason.lower()
        if any(k in r for k in ["网络","connection","timeout","超时","refused"]): return "single"
        if any(k in r for k in ["工具","tool","不支持","not found"]): return "method"
        if any(k in r for k in ["新信息","new info","发现"]): return "newinfo"
        return "full"

    def _replace(self, failed: Step, reason: str) -> Step:
        alt = self._generate_steps(f"失败({reason})换方案：{failed.description}", [], None)
        return alt[0] if alt else Step(
            step_id=str(uuid.uuid4())[:8],
            description=f"[重试]{failed.description}",
            tool=failed.tool, action=failed.action, expected=failed.expected)
