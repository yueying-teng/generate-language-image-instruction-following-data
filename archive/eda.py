# %%
import sys
sys.path.append("../")

import os
import glob
import json

import pandas as pd

from model_utils import read_file
from pretrain_595k import patterns

# %%
detail_23k_fp="../LLaVA-Instruct-150K/detail_23k.json"
complex_77k_fp="../LLaVA-Instruct-150K/complex_reasoning_77k.json"
conv_58k_fp="../LLaVA-Instruct-150K/conversation_58k.json"
pretrain_595k_fp = "../LLaVA-CC3M-Pretrain-595K/chat.json"

symbolic_rep_df = pd.read_pickle("../symbolic_representation_instruct_150k.pkl")
missing_bbox = pd.read_pickle("../instruct_150k_missing_bbox.pkl")["image"]

# %%
def get_prompt_msg_length(prompt_dir):
    length = 0
    cap_fps = glob.glob(f"{prompt_dir}/*_caps.txt")
    conv_fps = glob.glob(f"{prompt_dir}/*_conv.txt")

    sys_msg_path = f"{prompt_dir}/system_message.txt"
    length += len(read_file(sys_msg_path).split(" "))

    for cap_fp, conv_fp in zip(sorted(cap_fps), sorted(conv_fps)):
        cap = read_file(cap_fp)
        conv = read_file(conv_fp)
        length += len(cap.split(" ")) + len(conv.split(" "))

    return length

# %%
prompt_length = {}
for prompt_dir in sorted(os.listdir("../prompts")):
    # detail_description, complex_reasoning, conversation
    if prompt_dir != ".DS_Store":
        prompt_length[prompt_dir] = get_prompt_msg_length(os.path.join("../prompts", prompt_dir))

# %%
def get_ctx_length_stats(
    list_data_dict,
    symbolic_rep_df,
    missing_bbox,
    resp_type,
    prompt_length,
):

    image = [list_data_dict[i]["image"] for i in range(len(list_data_dict))]
    symbolic_rep_df = symbolic_rep_df[symbolic_rep_df["image"].isin(image)]

    if resp_type in ["complex_reasoning", "detail_description"]:
        symbolic_rep_df = symbolic_rep_df[~symbolic_rep_df["image"].isin(missing_bbox)]
        human_input = symbolic_rep_df.apply(
            lambda row: row["caption"] + "\n\n" + row["bbox"],
            axis=1,
        )
    else:
        human_input = symbolic_rep_df["caption"]

    ctx_length = human_input.apply(
        lambda row: len(row.split(" ")) + prompt_length[resp_type]
        )

    return ctx_length
# %%
def get_output_length_stats(
    list_data_dict,
    resp_type,
):
    output_length = []  # length of the generate output by the Assistant

    for i in range(len(list_data_dict)):
        if resp_type == "detail_description":
            for j, dic in enumerate(list_data_dict[i]["conversations"]):
                if j == 1:
                    output_length.append(len(dic["value"].split(" ")))

        elif resp_type == "complex_reasoning":
            cnt = 0
            for dic in list_data_dict[i]["conversations"]:
                cnt += len(dic["value"].split(" "))
            output_length.append(cnt)

        else:
            cnt = 0
            for dic in list_data_dict[i]["conversations"]:
                cnt += len(dic["value"].split(" "))
            output_length.append(cnt)

    return output_length


# %%
def get_length_stats(
    json_fp,
    symbolic_rep_df,
    missing_bbox,
    prompt_length,
    resp_type,
):
    print(json_fp)

    with open(json_fp) as f:
        list_data_dict = json.load(f)

    output_length = get_output_length_stats(list_data_dict, resp_type)
    ctx_length = get_ctx_length_stats(
        list_data_dict,
        symbolic_rep_df,
        missing_bbox,
        resp_type,
        prompt_length,
        )

    print("Assistant output length stats")
    print(pd.DataFrame(output_length).describe())

    print("Context length stats")
    print(pd.DataFrame(ctx_length).describe())

# %%
get_length_stats(
    json_fp=detail_23k_fp,
    symbolic_rep_df=symbolic_rep_df,
    missing_bbox=missing_bbox,
    prompt_length=prompt_length,
    resp_type="detail_description",
)

"""
                  0
count  23240.000000
mean     104.855164
std       18.721161
min       45.000000
25%       93.000000
50%      104.000000
75%      116.000000
max      205.000000
                  0
count  23240.000000
mean     960.509036
std       31.074869
min      916.000000
25%      939.000000
50%      953.000000
75%      974.000000
max     1276.000000
"""
# %%
get_length_stats(
    json_fp=complex_77k_fp,
    symbolic_rep_df=symbolic_rep_df,
    missing_bbox=missing_bbox,
    prompt_length=prompt_length,
    resp_type="complex_reasoning",
)
"""
                  0
count  76643.000000
mean     130.557507
std       37.444818
min       17.000000
25%      106.000000
50%      123.000000
75%      145.000000
max      348.000000
                  0
count  76276.000000
mean     895.805535
std       31.055056
min      857.000000
25%      874.000000
50%      885.000000
75%      908.000000
max     1250.000000
"""

# %%
get_length_stats(
    json_fp=conv_58k_fp,
    symbolic_rep_df=symbolic_rep_df,
    missing_bbox=missing_bbox,
    prompt_length=prompt_length,
    resp_type="conversation",
)

"""
                  0
count  56681.000000
mean     260.956052
std      112.417685
min        9.000000
25%      163.000000
50%      265.000000
75%      351.000000
max      708.000000
                  0
count  56681.000000
mean    1178.147474
std        6.264194
min     1165.000000
25%     1174.000000
50%     1177.000000
75%     1181.000000
max     1347.000000
"""

# %%
# pretrain_595k_fp
with open(detail_23k_fp) as f:
    list_data_dict = json.load(f)

instructions = set()
for i in range(len(list_data_dict)):
    instruction = list_data_dict[i]["conversations"][0]["value"]

    if instruction[:8] == patterns[0]:  # image first
        instructions.add(instruction.split(patterns[0])[-1])
    else:  # image last
        instructions.add(instruction.split(patterns[1])[0])

print(instructions)
