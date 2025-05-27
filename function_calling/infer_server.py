from qwen_agent.agents import Assistant
# from qwen_agent.gui import WebUI
import json
import re
from loguru import logger
from tqdm import tqdm
from math_verify import parse

# Define LLM
llm_cfg = {
    'model': 'models/Qwen3-0.6B',

    # Use a custom endpoint compatible with OpenAI API:
    'model_server': 'http://localhost:8000/v1',  # api_base
    'api_key': 'EMPTY',

    # Other parameters:
    'generate_cfg': {
            # Add: When the response content is `<think>this is the thought</think>this is the answer;
            # Do not add: When the response has been separated by reasoning_content and content.
            'thought_in_content': True,
        },
}

# Define Tools
tools = [
    {
        "mcpServers": {
            "calculator": {
                "command": "python",
                "args": ["-m", "mcp_server_calculator"]
            }
        }
    },
]

system_prompt = "You are a helpful assistant. Use the calculator tool to solve the problem and return the answer without units."

test_json_path = "data/test.json"

# Define Agent
bot = Assistant(llm=llm_cfg, function_list=tools, system_message=system_prompt)

def test():
    messages = [
        {'role': 'user', 'content': '红星玩具厂的一个生产小组生产一批玩具。原计划每天生产45件，4天做完。实际3天就做完成了任务。实际每天比原计划每天多做多少件玩具?'}
    ]
    for responses in bot.run(messages=messages):
        pass
    print(responses)

def parse_answer(content):
    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
    else:
        answer_text = content
    answer_parsed = parse(
        answer_text,
        extraction_mode="first_match",
    )
    return answer_parsed


def main():
    with open(test_json_path, 'r', encoding='utf-8') as file:
        test_data = json.load(file)

    with open("submit_tool_call.csv", 'w', encoding='utf-8') as file:
        for idx, row in enumerate(tqdm(test_data)):
            input_value = row['question']
            id = row['id']

            messages = [
                {"role": "user", "content": f"{input_value} /no_think"},
            ]
            logger.info(input_value)
            try:
                for responses in bot.run(messages=messages):
                    pass
                logger.info(responses)
                for response in responses[::-1]:
                    if response['role'] == 'assistant':
                        content = response['content']
                        answer_parsed = parse_answer(content)
                        answer_parsed = answer_parsed[0] if len(answer_parsed) > 1 else ""
                        file.write(f"{id},{answer_parsed}\n")
                        break
                logger.info(f"ID: {id}, Answer: {answer_parsed}")
            except Exception as e:
                logger.error(e)

        logger.info("All done!")

if __name__ == "__main__":
    # main()
    # WebUI(bot).run()
    test()
