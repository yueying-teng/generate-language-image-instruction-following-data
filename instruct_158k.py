# %%
import json

import pandas as pd
from langchain.llms import LlamaCpp

from pretrain_595k import replace_instruction, patterns, save_to_json
from model_utils import get_llm_chains

#%%
symbolic_rep_df = pd.read_pickle("symbolic_representation_instruct_150k.pkl")
missing_bbox = pd.read_pickle("instruct_150k_missing_bbox.pkl")["image"]

# %%
model_path = "./models/mistral-7b-instruct-v0.1.Q4_0.gguf"

# initialize LlamaCpp LLM model
# n_gpu_layers, n_batch, and n_ctx are for GPU support.
# When not set, CPU will be used.
# set 1 for Mac m2, and higher numbers based on your GPU support
llm = LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        repeat_penalty=1.2,
        top_p=1.,
        top_k=40,
        max_tokens=256,
        seed=2023,
        # n_gpu_layers=1,
        # n_batch=512,
        n_ctx=4096,
        verbose=True,
        )

llm_chains = get_llm_chains(llm)

# %%
def update_detail_23k(
    detail_23k_fp,
    replacing_instructions,
    symbolic_rep_df,
    llm_chains,
    test_first_k=2,  # update the entire detail_23k data using float("-inf")
    output_fp="updated_detail_23k.json",
):

    list_data_dict = json.load(open(detail_23k_fp, "r"))
    list_data_dict = replace_instruction(list_data_dict, replacing_instructions, patterns)

    for i in range(min(test_first_k, len(list_data_dict))):
        image = list_data_dict[i]["image"]
        symbolic_rep = symbolic_rep_df[symbolic_rep_df["image"] == image].iloc[0]

        human_input = symbolic_rep["caption"] + "\n\n" + symbolic_rep["bbox"]
        response = llm_chains["detail_description"].invoke({"input": human_input})

        list_data_dict[i]["conversations"][1]["value"] = response.split("AI:")[-1].strip().rstrip()

    save_to_json(list_data_dict, output_fp)

# %%
replacing_instructions = [
    "testing 01",
    "testing 02",
    "testing 03",
]

update_detail_23k(
    detail_23k_fp="LLaVA-Instruct-150K/detail_23k.json",
    replacing_instructions=replacing_instructions,
    symbolic_rep_df=symbolic_rep_df,
    llm_chains=llm_chains,
)


# %%
def remove_image_with_missing_bbox(list_data_dict, missing_bbox):
    images = [list_data_dict[i]["image"] for i in range(len(list_data_dict))]
    missing_bbox_indices = []
    for i in range(len(missing_bbox)):
        try:
            missing_bbox_indices.append(images.index(missing_bbox.iloc[i]))
        except ValueError:
            continue

    for i in missing_bbox_indices:
        list_data_dict.pop(i)

    return list_data_dict


def update_complex_77k(
    complex_77k_fp,
    symbolic_rep_df,
    llm_chains,
    missing_bbox,
    test_first_k=2,  # update the entire complex_77k data using float("-inf")
    output_fp="updated_complex_77k.json",
):
    list_data_dict = json.load(open(complex_77k_fp, "r"))
    # remove data_dict without coco bounding box annotations
    list_data_dict = remove_image_with_missing_bbox(list_data_dict, missing_bbox)

    for i in range(min(test_first_k, len(list_data_dict))):
        image = list_data_dict[i]["image"]
        symbolic_rep = symbolic_rep_df[symbolic_rep_df["image"] == image].iloc[0]

        human_input = symbolic_rep["caption"] + "\n\n" + symbolic_rep["bbox"]
        response = llm_chains["complex_reasoning"].invoke({"input": human_input})

        qa = response.split("Question:")[-1].split("===")
        question = qa[0].rstrip("\n").strip("\n").strip().rstrip()
        answer = qa[-1].split("Answer:")[-1].strip("\n")

        _ = replace_instruction([list_data_dict[i]], [question], patterns)

        list_data_dict[i]["conversations"][1]["value"] = answer

    save_to_json(list_data_dict, output_fp)

# %%
update_complex_77k(
    complex_77k_fp="LLaVA-Instruct-150K/complex_reasoning_77k.json",
    symbolic_rep_df=symbolic_rep_df,
    llm_chains=llm_chains,
    test_first_k=2,  # update the entire complex_77k data using float("-inf")
    output_fp="updated_complex_77k.json",
)

# %%
def update_conv_58k(
    conv_58k_fp,
    symbolic_rep_df,
    llm_chains,
    test_first_k=2,  # update the entire conv_58k data using float("-inf")
    output_fp="updated_conv_58k.json",
):
    list_data_dict = json.load(open(conv_58k_fp, "r"))

    for i in range(min(test_first_k, len(list_data_dict))):
        image = list_data_dict[i]["image"]
        symbolic_rep = symbolic_rep_df[symbolic_rep_df["image"] == image].iloc[0]

        human_input = symbolic_rep["caption"]
        response = llm_chains["conversation"].invoke({"input": human_input})

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
update_conv_58k(
    conv_58k_fp="LLaVA-Instruct-150K/conversation_58k.json",
    symbolic_rep_df=symbolic_rep_df,
    llm_chains=llm_chains,
    test_first_k=2,  # update the entire conv_58k data using float("-inf")
    output_fp="updated_conv_58k.json",
)

