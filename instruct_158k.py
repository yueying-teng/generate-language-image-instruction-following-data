# %%
import json
import sys
import pandas as pd
from langchain.llms import LlamaCpp

from model_utils import get_llm_chains

#%%
symbolic_rep_df = pd.read_pickle("symbolic_representation_instruct_150k.pkl")
missing_bbox = pd.read_pickle("instruct_150k_missing_bbox.pkl")["image"]

llm = LlamaCpp(
        model_path="./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
        temperature=0.69,
        repeat_penalty=1.24,
        top_p=0.98, #  0.98,
        top_k=90, # 50
        max_tokens=260, #260 detail_description, 350 for coomples_reasoning, 710 for conv
        seed=2023,
        n_gpu_layers=33,  # "Number of layers to be loaded into gpu memory
        n_batch=4200, # Number of tokens to process in parallel. Should be a number between 1 and n_ctx
        n_ctx=6200,  # Token context window
        verbose=False,
        )

llm_chains = get_llm_chains(llm)

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

# %%
def update_finetuning_data(
    json_fp,
    resp_type,
    symbolic_rep_df,
    llm_chains,
    test_first_k,  # update all the data using sys.maxsize
    output_fp,
    missing_bbox,
):
    assert resp_type in [
        "complex_reasoning", "detail_description", "conversation"
    ], \
        "resp_type must be in [complex_reasoning, detail_description, conversation]"

    with open(json_fp) as f:
        list_data_dict = json.load(f)

    if resp_type == "complex_reasoning":
        # remove data_dict without coco bounding box annotations
        list_data_dict = remove_image_with_missing_bbox(list_data_dict, missing_bbox)

    size = min(len(list_data_dict), test_first_k)

    if resp_type in ["complex_reasoning", "detail_description"]:
        symbolic_rep_df["human_input"] = symbolic_rep_df.apply(
                lambda row: str(row["caption"]) + "\n\n" + str(row["bbox"]),
                axis=1,
            )
    else:
        symbolic_rep_df["human_input"] = symbolic_rep_df["caption"]

    images = [list_data_dict[i]["image"] for i in range(size)][:size]
    responses = []

    for img in images:
        print(img)
        human_input = symbolic_rep_df[symbolic_rep_df["image"] == img].iloc[0]["human_input"]
        res = llm_chains[resp_type].invoke({"input": human_input})
        print(res)
        print("\n")
        responses.append(res)

    df = pd.DataFrame({"image": images, "response": responses})
    df.to_pickle(output_fp)


# %%
update_finetuning_data(
    json_fp="LLaVA-Instruct-150K/detail_23k.json",
    resp_type="detail_description",
    symbolic_rep_df=symbolic_rep_df,
    llm_chains=llm_chains,
    test_first_k=sys.maxsize,
    output_fp="generated_data/raw_detail_23k.pkl",
    missing_bbox=missing_bbox,
)


# %%
update_finetuning_data(
    json_fp="LLaVA-Instruct-150K/complex_reasoning_77k.json",
    resp_type="complex_reasoning",
    symbolic_rep_df=symbolic_rep_df,
    llm_chains=llm_chains,
    test_first_k=sys.maxsize,
    output_fp="generated_data/raw_complex_reasoning_77k.pkl",
    missing_bbox=missing_bbox,
)

# %%
update_finetuning_data(
    json_fp="LLaVA-Instruct-150K/conversation_58k.json",
    resp_type="conversation",
    symbolic_rep_df=symbolic_rep_df,
    llm_chains=llm_chains,
    test_first_k=sys.maxsize,
    output_fp="generated_data/raw_conversation_58k.pkl",
    missing_bbox=missing_bbox,
)
