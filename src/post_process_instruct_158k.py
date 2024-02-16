# %%
import json
import pandas as pd

from pretrain_595k import replace_instruction, patterns, save_to_json
from instruct_158k import remove_image_with_missing_bbox
from data_utils import replacing_instructions_detailed

#%%
detail_23k_raw_resp_fp = "generated_data/raw_detail_23k.pkl"
missing_bbox = pd.read_pickle("symbolic_rep_data/instruct_150k_missing_bbox.pkl")["image"]

# %%
def load_post_processing_data(
    json_fp,
    raw_resp_fp,
):

    with open(json_fp) as f:
        list_data_dict = json.load(f)

    raw_resp_df = pd.read_pickle(raw_resp_fp)

    return list_data_dict, raw_resp_df
# %%
def post_process_detail_23k(
    json_fp,
    replacing_instructions,
    raw_resp_fp,
    output_fp,
):
    list_data_dict, raw_resp_df = load_post_processing_data(json_fp, raw_resp_fp)

    list_data_dict = replace_instruction(list_data_dict, replacing_instructions, patterns)

    for i in range(len(list_data_dict)):
        image = list_data_dict[i]["image"]
        response = raw_resp_df[raw_resp_df["image"] == image]["response"].iloc[0]

        list_data_dict[i]["conversations"][1]["value"] = response.split("AI:")[-1].strip().rstrip()

    save_to_json(list_data_dict, output_fp)

# %%
post_process_detail_23k(
    json_fp="LLaVA-Instruct-150K/detail_23k.json",
    raw_resp_fp="generated_data/raw_detail_23k.pkl",
    replacing_instructions=replacing_instructions_detailed,
    output_fp="generated_data/detail_23k.json",
)

# %%
def post_process_complex_77k(
    json_fp,
    raw_resp_fp,
    output_fp,
    missing_bbox,
):
    list_data_dict, raw_resp_df = load_post_processing_data(json_fp, raw_resp_fp)

    # remove data_dict without coco bounding box annotations
    list_data_dict = remove_image_with_missing_bbox(list_data_dict, missing_bbox)

    for i in range(len(list_data_dict)):
        image = list_data_dict[i]["image"]
        response = raw_resp_df[raw_resp_df["image"] == image]["responses"].iloc[0]

        qa = response.split("Question:")[-1].split("===")
        question = qa[0].rstrip("\n").strip("\n").strip().rstrip()
        answer = qa[-1].split("Answer:")[-1].strip("\n")

        _ = replace_instruction([list_data_dict[i]], [question], patterns)

        list_data_dict[i]["conversations"][1]["value"] = answer

    save_to_json(list_data_dict, output_fp)

# %%
post_process_complex_77k(
    json_fp="LLaVA-Instruct-150K/complex_reasoning_77k.json",
    raw_resp_fp="generated_data/raw_complex_reasoning_77k.pkl",
    output_fp="generated_data/complex_reasoning_77k.json",
    missing_bbox=missing_bbox,
)
# %%
def post_process_conv_58k(
    json_fp,
    raw_resp_fp,
    output_fp,
):
    list_data_dict, raw_resp_df = load_post_processing_data(json_fp, raw_resp_fp)

    for i in range(len(list_data_dict)):
        image = list_data_dict[i]["image"]
        response = raw_resp_df[raw_resp_df["image"] == image]["responses"].iloc[0]

        sections = response.split('===')
        qa_list = []
        for j, section in enumerate(sections):
            if j % 2 == 0:
                q = section.split("Question:")[-1].strip("\n").rstrip("\n").strip().rstrip()
                if j == 0:
                    replace_instruction([list_data_dict[i]], [q], patterns)
                    qa_list.append(list_data_dict[i]["conversations"][0])
                else:
                    qa_list.append({"from": "human", "value": q})
            else:
                a = section.split("Answer:")[-1].strip("\n").rstrip("\n").strip().rstrip()
                qa_list.append({"from": "gpt", "value": a})

        list_data_dict[i]["conversations"] = qa_list

    save_to_json(list_data_dict, output_fp)

# %%
post_process_conv_58k(
    json_fp="LLaVA-Instruct-150K/conversation_58k.json",
    raw_resp_fp="generated_data/raw_conversation_58k.pkl",
    output_fp="generated_data/conversation_58k.json",
)
