"""
Channel Observer — 从 supervisor::events 收集灵光任务事件，注入到 LLM 上下文。

订阅流程：
  1. subscribe() 到 supervisor::events TopicChannel，获取待处理事件
  2. 立即 unsubscribe()（只取一次快照，不持续监听）
  3. 返回事件列表供 Hermes 注入上下文

用法：
    from agent.channel_observer import collect_supervisor_events, format_events_for_llm
    events = collect_supervisor_events(session_id="hermes-main")
    ctx = format_events_for_llm(events)
    # ctx = "[灵光任务状态]\n  - [task_completed] ..."

P1 自动错误积累：
    from agent.channel_observer import format_events_for_llm
    ctx = format_events_for_llm(events)  # 自动读取错误模式并注入
"""

import logging
import time
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# 订阅用的 consumer ID前缀，避免和真实订阅冲突
_CONSUMER_PREFIX = "hermes-observer-"


def collect_supervisor_events(
    session_id: str = "hermes-main",
    timeout: float = 0.3,
    max_events: int = 20,
) -> List[Dict[str, Any]]:
    """
    收集 supervisor::events Channel 上积累的所有事件。

    实现方式：订阅 → 收集所有可用事件 → 退订。
    TopicChannel 的 subscribe(include_last=True) 会把历史事件和新事件都返回，
    但我们不知道上次收集到哪条，所以用另一种方式：
    subscribe 一个临时消费者，立即 drain 其 queue，然后退订。

    Args:
        session_id: Hermes 主循环的 session_id（用于 Channel 隔离）
        timeout: 每个 receive() 的等待时间（秒）
        max_events: 最多收集多少条

    Returns:
        事件列表，每项是 dict：type / task_id / platform / instruction /
        timestamp / result / error / duration_ms
    """
    try:
        from agent.channels import get_channel, TopicChannel
    except Exception as exc:
        logger.debug("[channel-observer] cannot import channels: %s", exc)
        return []

    channel_name = "supervisor::events"
    consumer_id = f"{_CONSUMER_PREFIX}{int(time.time() * 1000)}"
    events: List[Dict[str, Any]] = []

    try:
        ch: TopicChannel = get_channel(
            channel_name,
            channel_type="topic",
            session_id=session_id,
        )
    except Exception as exc:
        logger.debug("[channel-observer] get_channel failed: %s", exc)
        return []

    # 用 subscribe + drain 方式收集事件：
    # 1. subscribe 一个临时消费者，include_last=True 取最近一条
    try:
        ch.subscribe(callback=None, include_last=True, consumer_id=consumer_id)
    except Exception as exc:
        logger.debug("[channel-observer] subscribe failed: %s", exc)
        return []

    # 2. 用 Channel.receive() drain 所有可用事件（每条最多等 timeout 秒）
    drained = 0
    while drained < max_events:
        try:
            event = ch.receive(timeout=timeout)
            if event is None:
                break
            events.append(event)
            drained += 1
        except Exception:
            break

    # 3. 退订，清理临时消费者
    try:
        ch.unsubscribe(consumer_id)
    except Exception:
        pass

    if events:
        logger.debug(
            "[channel-observer] collected %d event(s) from %s",
            len(events),
            channel_name,
        )

    return events


def format_events_for_llm(events: List[Dict[str, Any]]) -> str:
    """
    把事件列表格式化为 LLM 可读的文本片段，注入到 system prompt 或 user message。

    格式：
        [灵光任务状态]
        - [task_completed] task_id=xxx, platform=miaoshou, duration_ms=3049
          指令: "采集马来西亚热销耳机" → 成功
          评分: overall=7.2, accuracy=8.0, completeness=6.5, credibility=8.0, actionability=10.0
          弱点: ["Completeness < 50%"]
          决策: retry (质量尚可，建议重试提升完整度)
          结果: "..."
        - [task_timeout] task_id=xxx, platform=tiktok_sea
          指令: "采集TikTok热销榜" → 超时
        ...
    """
    if not events:
        return ""

    lines = ["[灵光任务状态]"]

    # ── P1: 注入错误历史上下文（来自 error_accumulator）──────────────
    try:
        from agent.error_accumulator import get_error_context
        error_patterns = get_error_context(max_warn=3)
        if error_patterns:
            lines.append("")
            lines.append("[⚠️ 历史错误模式 — 请优先处理]")
            for ep in error_patterns:
                level_icon = "🚨" if ep["level"] == "suppress" else "⚠️"
                lines.append(
                    f"  {level_icon} {ep['suggestion']} "
                    f"(累计 {ep['count']} 次, 示例错误: {ep['examples'][0]['msg'][:80] if ep['examples'] else 'N/A'})"
                )
    except Exception:
        pass  # error_accumulator 未安装或无数据，忽略
    # ─────────────────────────────────────────────────────────────────

    for ev in events[-10:]:  # 最多显示最近10条
        ev_type = ev.get("type", "?")
        tid = ev.get("task_id", "?")
        platform = ev.get("platform", "?")
        instruction = ev.get("instruction", "")
        duration = ev.get("duration_ms")
        result = ev.get("result")
        error = ev.get("error")
        ts = ev.get("timestamp")
        ts_str = ""
        if ts:
            try:
                ts_str = time.strftime("%H:%M:%S", time.localtime(ts))
            except Exception:
                ts_str = ""

        prefix = f"[{ts_str}] " if ts_str else ""
        lines.append(f"{prefix}[{ev_type}] task_id={tid}, platform={platform}")
        if instruction:
            lines.append(f"  指令: {instruction}")
        if duration:
            lines.append(f"  耗时: {duration}ms")

        # Quality score (P0 feature)
        qs = ev.get("quality_score")
        if qs:
            lines.append(
                f"  评分: overall={qs['overall']}, "
                f"accuracy={qs['accuracy']}, completeness={qs['completeness']}, "
                f"credibility={qs['credibility']}, actionability={qs['actionability']}"
            )
            if qs.get("weaknesses"):
                lines.append(f"  弱点: {qs['weaknesses']}")
            decision = qs.get("retry_decision", "")
            if decision and decision != "accept":
                lines.append(f"  ⚠️ 决策: {decision} (评分={qs['overall']})")
                # ⚡ 强制行动指令 — 模型必须遵守，不能忽略评分建议
                lines.append(f"  → 系统指令：必须执行 {decision.upper()}，不得跳过或忽略此评分建议")
            elif decision == "accept":
                lines.append(f"  ✅ 决策: accept")

        # P1: per-event error pattern (from error_accumulator)
        ep = ev.get("error_pattern")
        if ep:
            level_icon = "🚨" if ep.get("level") == "suppress" else "⚠️"
            lines.append(f"  {level_icon} 错误累计: {ep.get('count')}次 → {ep.get('suggestion', '')[:150]}")

        if result:
            text = result.get("text", "")
            if text:
                lines.append(f"  结果: {text[:200]}")
        if error:
            lines.append(f"  错误: {error}")

    return "\n".join(lines)
