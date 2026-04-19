#RULE-BASED POLICY
unsafe_keywords = {
    "self-harm": {
        "keyword": ["自杀", "想死", "自我了断", "轻生"],
        "action": "support"
    },
    "violence": {
        "keyword": ["杀", "杀人", "伤害"],
        "action": "block"
    },
    "sexual": {
        "keyword": ["强奸", "色情"],
        "action": "block"
    }
}

def is_unsafe(user_input):
    text = user_input.lower()
    for category, value in unsafe_keywords.items():
        keywords = value["keyword"]
        action = value["action"]

        for keyword in keywords:
            if keyword in text:
                return True, category, action
    return False, None, None