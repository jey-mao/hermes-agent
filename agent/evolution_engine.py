"""
Hermes v0.4 — 进化层 EvolutionEngine
基于 Self-Improving-Agent 的 prompt_history 持久化 + 相似度检索。
"""
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


TRAJ_DIR = Path.home() / ".hermes" / "v04_trajectories"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)
TRAJ_FILE = TRAJ_DIR / "trajectories.jsonl"


@dataclass
class TrajectoryRecord:
    ts: int
    task_sig_hash: str
    task_sig_domain: str
    task_sig_action: str
    task_sig_constraints: str
    prompt: str
    score: float
    evaluation: dict
    steps: list
    duration_ms: int
    result_summary: str
    retry_count: int = 0
    replan_count: int = 0
    source: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)


class EvolutionEngine:
    def __init__(self, trajectory_file: Path = TRAJ_FILE):
        self.file = trajectory_file
        TRAJ_DIR.mkdir(parents=True, exist_ok=True)
        if not self.file.exists():
            self.file.write_text("")

    def record(self, record: TrajectoryRecord):
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def record_simple(self, task_sig_hash: str, task_sig_domain: str,
                      task_sig_action: str, task_sig_constraints: str,
                      prompt: str, score: float, evaluation: dict,
                      steps: list, duration_ms: int, result_summary: str,
                      retry_count=0, replan_count=0):
        self.record(TrajectoryRecord(
            ts=int(time.time()),
            task_sig_hash=task_sig_hash, task_sig_domain=task_sig_domain,
            task_sig_action=task_sig_action, task_sig_constraints=task_sig_constraints,
            prompt=prompt, score=score, evaluation=evaluation,
            steps=steps, duration_ms=duration_ms, result_summary=result_summary,
            retry_count=retry_count, replan_count=replan_count))

    def find_similar(self, task_sig, top_k: int = 5) -> Optional[dict]:
        """查找同类任务的历史最佳方案"""
        candidates = []
        if not self.file.exists(): return None
        with open(self.file, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    rec = json.loads(line)
                    if rec.get("task_sig_domain") == task_sig.domain and                        rec.get("task_sig_hash","")[:4] == task_sig.hash[:4]:
                        candidates.append(rec)
                except Exception:
                    pass
        if not candidates:
            with open(self.file, encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        rec = json.loads(line)
                        if rec.get("task_sig_domain") == task_sig.domain:
                            candidates.append(rec)
                    except Exception:
                        pass
        if not candidates: return None
        return max(candidates, key=lambda r: r.get("score", 0))

    def get_best_prompt(self, task_sig, threshold: float = 7.0) -> Optional[str]:
        best = self.find_similar(task_sig)
        if best and best.get("score", 0) >= threshold:
            return best.get("prompt")
        return None

    def stats(self) -> dict:
        records = []
        if self.file.exists():
            with open(self.file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try: records.append(json.loads(line))
                        except: pass
        if not records:
            return {"total":0,"avg_score":0,"best_score":0,"domains":{}}
        scores = [r.get("score",0) for r in records]
        domains = {}
        for r in records: domains[r.get("task_sig_domain","?")] = domains.get(r.get("task_sig_domain","?"),0)+1
        return {"total":len(records),"avg_score":round(sum(scores)/len(scores),2),
                "best_score":round(max(scores),2),"worst_score":round(min(scores),2),"domains":domains}
