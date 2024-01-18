# %%
import json

from pretrain_595k import replace_instruction, patterns
#%%
data_path = "../LLaVA-Instruct-150K"
fns = [
    "conversation_58k.json",
    "complex_reasoning_77k.json",
    "detail_23k.json",
]

data_path = "../LLaVA-Instruct-150K/detail_23k.json"
detail_list_data_dict = json.load(open(data_path, "r"))

# %%
detail_list_data_dict[0]["conversations"][1]["value"]
# %%

# %%

# # %%
# replacing_instructions = [
#     "testing 01",
#     "testing 02",
#     "testing 03",
# ]
# detail_list_data_dict = replace_instruction(detail_list_data_dict, replacing_instructions, patterns)

