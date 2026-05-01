from typing import Dict


def classify_risk(risk_score: float, phq9_item9_flag: bool) -> Dict[str, str]:
    if phq9_item9_flag:
        return {
            "risk_level": "CRITICAL",
            "explanation": "PHQ-9第9题提示存在自伤相关风险信号，建议立即联系专业支持。",
        }

    if risk_score > 0.6:
        level = "HIGH"
        explanation = "综合风险分较高，建议尽快进行专业心理咨询并启动稳定化干预。"
    elif risk_score > 0.3:
        level = "MEDIUM"
        explanation = "当前存在中等程度心理负担，建议开展结构化自助与规律随访。"
    else:
        level = "LOW"
        explanation = "当前风险较低，建议持续记录情绪并保持规律作息。"

    return {"risk_level": level, "explanation": explanation}
