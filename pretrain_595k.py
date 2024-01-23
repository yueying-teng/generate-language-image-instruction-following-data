# %%
import os
import json
import random

# %%
patterns = [
    "<image>\n",  # image first
    "\n<image>",  # image last
            ]


replacing_instructions = [
    "testing 01",
    "testing 02",
    "testing 03",
]

def replace_instruction(list_data_dict, replacing_instructions, patterns):
    for i in range(len(list_data_dict)):
        instruction = list_data_dict[i]["conversations"][0]["value"]
        new_instruction = random.choice(replacing_instructions)
        if instruction[:8] == patterns[0]:  # image first
            list_data_dict[i]["conversations"][0]["value"] = patterns[0] + new_instruction
        else:  # image last
            list_data_dict[i]["conversations"][0]["value"] = new_instruction + patterns[1]

    return list_data_dict


def save_to_json(data, output_fp):
    with open(output_fp, "w") as f:
        json.dump(data, f)

# %%
if __name__ == "__main__":
    data_path = "LLaVA-CC3M-Pretrain-595K/chat.json"
    list_data_dict = json.load(open(data_path, "r"))

    list_data_dict = replace_instruction(list_data_dict, replacing_instructions, patterns)

    save_to_json(list_data_dict, "generate_data/chat.json")
