from typing import Dict, List


SAFETY_DISCLAIMER = "本系统为心理筛查工具，非医学诊断。"


def generate_intervention(risk_level: str, vector: Dict[str, float]) -> Dict[str, object]:
    base = {
        "disclaimer": SAFETY_DISCLAIMER,
        "risk_level": risk_level,
        "actions": [],
        "restrictions": [],
    }

    if risk_level == "LOW":
        base["actions"] = [
            "持续每日情绪记录与睡眠打卡",
            "使用AI陪伴进行低负荷情绪表达",
            "每周复测一次，观察状态趋势",
        ]
    elif risk_level == "MEDIUM":
        base["actions"] = [
            "执行CBT思维记录（触发事件-自动想法-证据平衡）",
            "每天2次腹式呼吸训练（每次3-5分钟）",
            "建立作息稳定计划并减少夜间高刺激活动",
        ]
    elif risk_level == "HIGH":
        base["actions"] = [
            "优先启动明确心理调节方案（睡眠、压力、社交支持）",
            "尽快预约学校心理中心/医院心理门诊",
            "安排可信任联系人共同监测近期状态变化",
        ]
        base["restrictions"] = ["减少轻松安慰型回复，优先结构化干预建议"]
    else:  # CRITICAL
        base["actions"] = [
            "立即联系身边可信任的人并保持陪伴",
            "尽快联系当地心理危机干预热线或120/110等紧急服务",
            "前往就近医院急诊精神心理科进行评估",
        ]
        base["hotlines"] = [
            "全国心理援助热线：12356（请以当地最新公布为准）",
            "紧急情况：120 / 110",
        ]
        base["restrictions"] = ["禁止生成轻松安慰型AI内容，优先输出危机处置步骤"]

    if vector.get("sleep", 1) < 0.4:
        base.setdefault("focus", []).append("睡眠修复")
    if vector.get("stress", 0) > 0.6:
        base.setdefault("focus", []).append("压力管理")

    return base
