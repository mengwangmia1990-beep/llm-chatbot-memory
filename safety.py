#RULE-BASED POLICY
unsafe_keywords = {
    "self-harm": {
        "keyword": ["sucicide", "end myself", "kill myself"],
        "action": "support"
    },
    "violence": {
        "keyword": ["kill", "kill someone", "harm"],
        "action": "block"
    },
    "sexual": {
        "keyword": ["rape", "porn"],
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