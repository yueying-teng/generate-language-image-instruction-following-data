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
    "Describe the image in a nutshell.",
    "Give a brief summary of the given picture.",
    "Present a succinct overview of the photograph.",
    "Summarize the key visual elements of the image.",
    "Offer a short and clear description of the visual content in the image.",
    "Provide a concise explanation of what is shown in the photo.",
    "Share an abridged account of the scene depicted in the shown picture.",
    "Write a compact, clear interpretation of the picture.",
    "Relay a concise narrative representing the image presented.",
    "Render a condensed but informative summary of the picture.",
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

    save_to_json(list_data_dict, "generated_data/chat.json")
