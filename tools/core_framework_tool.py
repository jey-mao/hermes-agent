"""
core_framework — 框架级内置自我反思工具

每次 run_conversation 开始时自动调用（系统层面强制，非模型主动选择）。
输出直接注入消息上下文，影响模型推理。
"""

import json
import threading
from pathlib import Path
from typing import Optional

# ============================================================================
# Framework Data Storage
# ============================================================================

FRAMEWORK_VERSION = "1.0.0"
FRAMEWORK_FILE = Path.home() / ".hermes" / "core_framework.json"
FRAMEWORK_LOCK = threading.Lock()


def _load_framework() -> dict:
    """加载核心框架数据"""
    if FRAMEWORK_FILE.exists():
        try:
            return json.loads(FRAMEWORK_FILE.read_text())
        except Exception:
            pass
    return {
        "version": FRAMEWORK_VERSION,
        "evolution_log": [],
        "current_focus": [],
        "reflection_template": [
            "我上轮犯的具体错误是什么？",
            "这个错误背后的思维模式是什么？",
            "这次回答比上次改进了吗？改进了什么？",
        ],
        "last_turn_reflection": None,
        "last_turn_answer": None,
    }


def _save_framework(data: dict) -> None:
    """保存核心框架数据"""
    with FRAMEWORK_LOCK:
        FRAMEWORK_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ============================================================================
# Core Framework Logic
# ============================================================================

def run_core_framework(
    previous_answer: Optional[str] = None,
    current_message: str = "",
    session_db = None,
    current_session_id: str = None,
) -> dict:
    """
    执行核心框架三层反思。
    
    返回结构化结果，用于注入到模型上下文中。
    """
    framework = _load_framework()
    
    # --- 1. 自我反省模块 ---
    reflections = []
    questions = framework.get("reflection_template", [])
    last_answer = framework.get("last_turn_answer")
    
    if previous_answer and last_answer:
        reflections.append(f"【上轮回答摘要】{last_answer[:150]}")
        reflections.append("【本次反省】")
        for q in questions[:2]:
            reflections.append(f"  ▶ {q}")
    elif last_answer:
        reflections.append(f"【上轮回答摘要】{last_answer[:150]}")
        reflections.append("【本次反省】无新反省输入")
    else:
        reflections.append("【初始启动】核心框架首次激活，无历史反省")
    
    self_reflection_text = "\n".join(reflections)
    
    # --- 2. 经验召回模块（改为自动调用session_search + skill_view）---
    experience_parts = []
    
    # 2.1 自动调用session_search，召回相关历史
    try:
        from tools.session_search_tool import session_search
        if session_db is not None and current_message.strip():
            search_result = session_search(
                query=current_message[:200],
                limit=2,
                db=session_db,
                current_session_id=current_session_id,
            )
            if search_result and "无匹配" not in search_result and "没有找到" not in search_result:
                experience_parts.append(f"【历史召回-session_search】\n{search_result[:500]}")
            else:
                experience_parts.append("【历史召回-session_search】无相关历史")
        else:
            experience_parts.append("【历史召回-session_search】(db未传入或消息为空，跳过)")
    except Exception as e:
        experience_parts.append(f"【历史召回-session_search】调用失败: {str(e)[:100]}")
    
    # 2.2 自动调用skill_view，加载三个核心技能
    core_skills = ["self-cognition", "accumulation-protocol", "error-correction"]
    loaded_skills = []
    try:
        from tools.skills_tool import skill_view
        for skill_name in core_skills:
            try:
                skill_result = skill_view(name=skill_name, file_path=None, task_id="")
                if skill_result and len(skill_result) > 50:
                    loaded_skills.append(f"{skill_name}({len(skill_result)}字符)")
                else:
                    loaded_skills.append(f"{skill_name}(失败)")
            except Exception as e:
                loaded_skills.append(f"{skill_name}(异常)")
        experience_parts.append(f"【技能加载-skill_view】{', '.join(loaded_skills)}")
    except Exception as e:
        experience_parts.append(f"【技能加载-skill_view】调用失败: {str(e)[:100]}")
    
    # 2.3 保留旧的evolution_log作为补充
    experience_log = framework.get("evolution_log", [])
    if experience_log:
        recent_reflections = [e.get('reflection', '')[:80] for e in experience_log[-2:]]
        if recent_reflections:
            experience_parts.append(f"【反复犯错记录】{' | '.join(recent_reflections)}")
    
    experience_text = "\n".join(experience_parts) if experience_parts else "【经验召回】无可用数据"
    
    # --- 3. 进化验证模块 ---
    last_reflection = framework.get("last_turn_reflection")
    if last_reflection:
        evolution_text = f"【进化验证】上次反省：{last_reflection[:100]}"
    else:
        evolution_text = "【进化验证】无上次反省记录，无法比较"
    
    # 组装完整结果
    result_text = (
        f"{self_reflection_text}\n\n"
        f"{experience_text}\n\n"
        f"{evolution_text}\n\n"
        f"─── 核心框架 v{FRAMEWORK_VERSION} │ 系统级强制调用 │ 不可跳过 ───"
    )
    
    return {
        "version": FRAMEWORK_VERSION,
        "reflection": self_reflection_text,
        "experience": experience_text,
        "evolution": evolution_text,
        "result_text": result_text,
    }


def record_evolution(reflection_text: str, improvement_text: str = "") -> dict:
    """记录一次进化到框架数据中"""
    framework = _load_framework()
    
    entry = {
        "reflection": reflection_text,
        "improvement": improvement_text,
    }
    
    framework["evolution_log"].append(entry)
    
    # 只保留最近50条
    if len(framework["evolution_log"]) > 50:
        framework["evolution_log"] = framework["evolution_log"][-50:]
    
    _save_framework(framework)
    
    return {"status": "recorded", "total_entries": len(framework["evolution_log"])}


# ============================================================================
# Tool Handler (for registry.register)
# ============================================================================

def _handle_core_framework(args: dict, task_id: str = "", user_task: str = "") -> str:
    """工具处理器，供 Hermes 工具系统调用"""
    previous_answer = args.get("previous_answer")
    current_message = args.get("current_message", "")
    
    result = run_core_framework(
        previous_answer=previous_answer,
        current_message=current_message,
    )
    
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================================
# Registry Registration
# ============================================================================

CORE_FRAMEWORK_SCHEMA = {
    "name": "core_framework",
    "description": (
        "框架级内置自我反思工具。每次回答前系统自动调用，不可跳过。 "
        "包含三层反思：自我反省（上轮哪里错）、经验召回（相关经历）、进化验证（比上轮改进了吗）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "previous_answer": {
                "type": "string",
                "description": "上轮模型的回答内容，用于反省和进化验证",
            },
            "current_message": {
                "type": "string",
                "description": "当前收到的用户消息，用于经验召回匹配",
            },
        },
        "required": ["previous_answer", "current_message"],
    },
}


def _check_core_framework_requirements() -> bool:
    """检查 core_framework 是否可用（始终可用）"""
    return True


try:
    from tools.registry import registry as _registry
    _registry.register(
        name="core_framework",
        toolset="internal",
        schema=CORE_FRAMEWORK_SCHEMA,
        handler=_handle_core_framework,
        check_fn=_check_core_framework_requirements,
        is_async=False,
        emoji="⚙️",
        description="框架级内置自我反思工具（系统强制调用）",
    )
except ImportError:
    # 早期导入时 registry 尚未加载，静默跳过
    pass
