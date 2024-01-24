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
        repeat_penalty=1.2,
        top_p=0.99, #  0.98,
        top_k=40, # 50
        max_tokens=780,
        seed=2023,
        n_gpu_layers=33,  # "Number of layers to be loaded into gpu memory
        n_batch=4096, # Number of tokens to process in parallel. Should be a number between 1 and n_ctx
        n_ctx=4096,  # Token context window
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


def batch_predict(
    i,
    images,
    symbolic_rep_df,
    llm_chains,
    batch_size,
    max_concurrency,
    resp_type,
):
    cur_i = i * batch_size
    batch_images = images[cur_i: cur_i + batch_size]
    symbolic_rep = symbolic_rep_df[symbolic_rep_df["image"].isin(batch_images)]

    batch_data = [{"input": h} for h in symbolic_rep["human_input"]]

    responses = llm_chains[resp_type].batch(
        batch_data,
        config={"max_concurrency": max_concurrency}  # set the number of concurrent requests by using the max_concurrency parameter
    )

    return responses

# %%
def update_finetuning_data(
    json_fp,
    resp_type,
    symbolic_rep_df,
    llm_chains,
    batch_size,
    max_concurrency,
    test_first_k,  # update the data using sys.maxsize
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
    num_batches = size // batch_size

    if resp_type in ["complex_reasoning", "detail_description"]:
        symbolic_rep_df["human_input"] = symbolic_rep_df.apply(
                lambda row: str(row["caption"]) + "\n\n" + str(row["bbox"]),
                axis=1,
            )
    else:
        symbolic_rep_df["human_input"] = symbolic_rep_df["caption"]

    images = [list_data_dict[i]["image"] for i in range(size)]
    responses = []

    i = 0
    while i < num_batches:
        responses += batch_predict(
            i,
            images,
            symbolic_rep_df,
            llm_chains,
            batch_size,
            max_concurrency,
            resp_type,
        )

        i += 1
        cur_i = i * batch_size
        if i >= num_batches and images[cur_i: size] != []:
            responses += batch_predict(
                i,
                images,
                symbolic_rep_df,
                llm_chains,
                batch_size,
                max_concurrency,
                resp_type,
            )

    df = pd.DataFrame({"image": images, "responses": responses})
    df.to_pickle(output_fp)


# %%
update_finetuning_data(
    json_fp="LLaVA-Instruct-150K/detail_23k.json",
    resp_type="detail_description",
    symbolic_rep_df=symbolic_rep_df,
    llm_chains=llm_chains,
    batch_size=4,
    max_concurrency=2,
    test_first_k=2,  # update all data with sys.maxsize
    output_fp="generated_data/raw_detail_23k.pkl",
)

# %%

update_finetuning_data(
    json_fp="LLaVA-Instruct-150K/complex_reasoning_77k.json",
    resp_type="complex_reasoning",
    symbolic_rep_df=symbolic_rep_df,
    llm_chains=llm_chains,
    batch_size=4,
    max_concurrency=2,
    test_first_k=2,  # update all data with sys.maxsize
    output_fp="generated_data/raw_complex_reasoning_77k.pkl",
)

# %%
update_finetuning_data(
    json_fp="LLaVA-Instruct-150K/conversation_58k.json",
    resp_type="conversation",
    symbolic_rep_df=symbolic_rep_df,
    llm_chains=llm_chains,
    batch_size=4,
    max_concurrency=2,
    test_first_k=2,  # update all data with sys.maxsize
    output_fp="generated_data/raw_conversation_58k.pkl",
)
