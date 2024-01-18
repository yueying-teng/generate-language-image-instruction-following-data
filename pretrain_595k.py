# %%
import os
import json
import random

#%%
data_path = "../LLaVA-CC3M-Pretrain-595K/chat.json"
list_data_dict = json.load(open(data_path, "r"))

instructions = set()
for i in range(len(list_data_dict)):
    instruction = sorted(list_data_dict[i]["conversations"][0]["value"].split("<image>"), key=len, reverse=True)[0]
    instructions.add(sorted(instruction.split("\n"), key=len, reverse=True)[0])

print(len(instructions))
instructions

# %%
images = set()
for i in range(len(list_data_dict)):
    images.add(list_data_dict[i]["image"])

# %%
image_folder = "../LLaVA-CC3M-Pretrain-595K/images"
for img in os.listdir(image_folder):
    if img not in images:
        print(img)

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

for i in range(len(list_data_dict)):
    instruction = list_data_dict[i]["conversations"][0]["value"]
    new_instruction = random.choice(replacing_instructions)
    if instruction[:8] == patterns[0]:  # image first
        list_data_dict[i]["conversations"][0]["value"] = patterns[0] + new_instruction
    else:  # image last
        list_data_dict[i]["conversations"][0]["value"] = new_instruction + patterns[1]
    if i == 10:
        break

# %%
with open("../LLaVA-CC3M-Pretrain-595K/chat_instruction_replaced.json", "w") as f:
    json.dump(list_data_dict, f)
