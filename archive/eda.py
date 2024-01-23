# %%
import os
import json
import pandas as pd

from model_utils import read_file

# %%
detail_23k_fp="LLaVA-Instruct-150K/detail_23k.json"
complex_77k_fp="LLaVA-Instruct-150K/complex_reasoning_77k.json"
conv_58k_fp="LLaVA-Instruct-150K/conversation_58k.json"

symbolic_rep_df = pd.read_pickle("symbolic_representation_instruct_150k.pkl")
missing_bbox = pd.read_pickle("instruct_150k_missing_bbox.pkl")["image"]

sys_msg_length = {}
for prompt_dir in sorted(os.listdir("prompts")):
    # detail_description, complex_reasoning, conversation
    if prompt_dir != ".DS_Store":
        sys_msg_path = f"./prompts/{prompt_dir}/system_message.txt"
        sys_msg_length[prompt_dir] = len(read_file(sys_msg_path).split(" "))


# %%
def get_ctx_length_stats(
    list_data_dict,
    symbolic_rep_df,
    missing_bbox,
    json_type,
):

    image = [list_data_dict[i]["image"] for i in range(len(list_data_dict))]
    symbolic_rep_df = symbolic_rep_df[symbolic_rep_df["image"].isin(image)]

    if json_type in ["complex_reasoning", "detail_description"]:
        symbolic_rep_df = symbolic_rep_df[~symbolic_rep_df["image"].isin(missing_bbox)]
        human_input = symbolic_rep_df.apply(
            lambda row: row["caption"] + "\n\n" + row["bbox"],
            axis=1,
        )
    else:
        human_input = symbolic_rep_df["caption"]

    ctx_length = human_input.apply(
        lambda row: len(row.split(" ")) + sys_msg_length[json_type]
        )

    return ctx_length
# %%
def get_output_length_stats(
    list_data_dict,
    json_type,
):
    output_length = []  # length of the generate output by the Assistant

    for i in range(len(list_data_dict)):
        if json_type == "detail_description":
            for j, dic in enumerate(list_data_dict[i]["conversations"]):
                if j == 1:
                    output_length.append(len(dic["value"].split(" ")))

        elif json_type == "complex_reasoning":
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
    json_type,
):
    print(json_fp)
    list_data_dict = json.load(open(json_fp, "r"))

    output_length = get_output_length_stats(list_data_dict, json_type)
    ctx_length = get_ctx_length_stats(
        list_data_dict,
        symbolic_rep_df,
        missing_bbox,
        json_type,
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
    json_type="detail_description",
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
mean     280.509036
std       31.074869
min      236.000000
25%      259.000000
50%      273.000000
75%      294.000000
max      596.000000
"""
# %%
get_length_stats(
    json_fp=complex_77k_fp,
    symbolic_rep_df=symbolic_rep_df,
    missing_bbox=missing_bbox,
    json_type="complex_reasoning",
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
Context length stats
                  0
count  76276.000000
mean     320.805535
std       31.055056
min      282.000000
25%      299.000000
50%      310.000000
75%      333.000000
max      675.000000
"""

# %%
get_length_stats(
    json_fp=conv_58k_fp,
    symbolic_rep_df=symbolic_rep_df,
    missing_bbox=missing_bbox,
    json_type="conversation",
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
mean     267.147474
std        6.264194
min      254.000000
25%      263.000000
50%      266.000000
75%      270.000000
max      436.000000
"""

