import json
import config
import os
import rag
from main import generate_reply, system_message, set_trace

EVAL_DIR = config.EVAL_DIR
EVAL_DATA_FILE = os.path.join(EVAL_DIR, config.EVAL_DATA_FILE)

# load knowledge base
knowledge = rag.load_knowledge()

messages = [system_message]

with open(EVAL_DATA_FILE) as f:
    for line in f:
        data = json.loads(line)
        query = data["query"]
        print(query)
        reply, rag_result = generate_reply(query, knowledge, messages)
        set_trace(query, rag_result, reply)
        