from typing import Dict

SCALE_MAX = {
    "GAD7": 21,
    "PHQ9": 27,
    "PSS14": 56,
    "RSES": 40,
    "ISP": 66,
    "PSQI": 24,
}

PSS14_REVERSED = {4, 5, 6, 7, 9, 10, 13}
RSES_REVERSED = {3, 5, 8, 9, 10}


def _sum_with_reverse(items: Dict[int, int], reversed_items, max_item_score: int) -> int:
    total = 0
    for idx, raw in items.items():
        raw_value = int(raw)
        if idx in reversed_items:
            total += max_item_score - raw_value
        else:
            total += raw_value
    return total


def calculate_scale_scores(answers: Dict[str, Dict[int, int]]) -> Dict[str, int]:
    gad7_items = answers.get("GAD7", {})
    phq9_items = answers.get("PHQ9", {})
    pss14_items = answers.get("PSS14", {})
    rses_items = answers.get("RSES", {})
    isp_items = answers.get("ISP", {})
    psqi_items = answers.get("PSQI", {})

    gad7_score = sum(int(v) for v in gad7_items.values())
    phq9_score = sum(int(v) for v in phq9_items.values())
    pss14_score = _sum_with_reverse(pss14_items, PSS14_REVERSED, 4)
    rses_score = _sum_with_reverse(rses_items, RSES_REVERSED, 5)
    isp_score = sum(int(v) for v in isp_items.values())
    psqi_score = sum(int(v) for v in psqi_items.values())

    phq9_item9 = int(phq9_items.get(9, 0))
    phq9_item9_flag = phq9_item9 >= 1

    return {
        "gad7_score": gad7_score,
        "phq9_score": phq9_score,
        "pss14_score": pss14_score,
        "rses_score": rses_score,
        "interpersonal_score": isp_score,
        "psqi_score": psqi_score,
        "phq9_item9_flag": phq9_item9_flag,
    }


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def build_mental_vector(scores: Dict[str, int]) -> Dict[str, float]:
    anxiety = _clip01(scores.get("gad7_score", 0) / SCALE_MAX["GAD7"])
    depression = _clip01(scores.get("phq9_score", 0) / SCALE_MAX["PHQ9"])
    stress = _clip01(scores.get("pss14_score", 0) / SCALE_MAX["PSS14"])
    sleep = _clip01(1 - (scores.get("psqi_score", 0) / SCALE_MAX["PSQI"]))
    self_esteem = _clip01((scores.get("rses_score", 10) - 10) / 30)
    interpersonal = _clip01(scores.get("interpersonal_score", 0) / SCALE_MAX["ISP"])

    return {
        "anxiety": anxiety,
        "depression": depression,
        "stress": stress,
        "sleep": sleep,
        "self_esteem": self_esteem,
        "interpersonal": interpersonal,
    }
