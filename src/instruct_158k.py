# %%
import csv
import json
import sys
from datetime import datetime
import pandas as pd
from llama_cpp import Llama

from model_utils import SAMPLING_CONFIG, get_all_prompts

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
def complete(llm, human_input, prompts, sampling_config):
    llm.reset()
    resp = llm.create_chat_completion(
            messages = prompts + [
                {
                    "role": "user",
                    "content": human_input,
                },
            ],
            temperature=sampling_config["temperature"],
            repeat_penalty=sampling_config["repeat_penalty"],
            top_p=sampling_config["top_p"],
            top_k=sampling_config["top_k"],
            max_tokens=sampling_config["max_tokens"],
            seed=2023,
        )

    return resp["choices"][0]["message"]["content"]

# %%
def predict_and_save(
    llm,
    list_data_dict,
    symbolic_rep_df,
    final_prompts,
    test_first_k,
    resp_type,
    starting_img,
    output_fp,
    ):

    size = min(len(list_data_dict), test_first_k)
    images = [list_data_dict[i]["image"] for i in range(size)][:size]

    starting_img_idx = images.index(starting_img) if starting_img != -1 else 0

    with open(output_fp, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "response"])

        for img in images[starting_img_idx:]:
            row = [img]
            print(img)
            human_input = symbolic_rep_df[symbolic_rep_df["image"] == img].iloc[0]["human_input"]
            res = complete(
                llm,
                human_input,
                final_prompts[resp_type],
                SAMPLING_CONFIG[resp_type],
                )
            print(res)
            print("\n")
            row.append(res)
            writer.writerow(row)

# %%
def update_finetuning_data(
    llm,
    json_fp,
    resp_type,
    symbolic_rep_df,
    test_first_k,
    output_fp,
    missing_bbox,
    final_prompts,
    starting_img=-1,
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

    if resp_type in ["complex_reasoning", "detail_description"]:
        symbolic_rep_df["human_input"] = symbolic_rep_df.apply(
                lambda row: str(row["caption"]) + "\n\n" + str(row["bbox"]),
                axis=1,
            )
    else:
        symbolic_rep_df["human_input"] = symbolic_rep_df["caption"]

    predict_and_save(
        llm,
        list_data_dict,
        symbolic_rep_df,
        final_prompts,
        test_first_k,
        resp_type,
        starting_img,
        output_fp,
    )

# %%
if __name__ == "__main__":
    symbolic_rep_df = pd.read_pickle("symbolic_rep_data/symbolic_representation_instruct_150k.pkl")
    missing_bbox = pd.read_pickle("symbolic_rep_data/instruct_150k_missing_bbox.pkl")["image"]

    llm = Llama(
        model_path="./models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf",
        n_batch=6200, # Number of tokens to process in parallel. Should be a number between 1 and n_ctx
        n_ctx=6200,
        n_gpu_layers=33,
        chat_format="mistral-instruct",
        verbose=False,
    )

    final_prompts = get_all_prompts("prompts")
    test_first_k = sys.maxsize

    print("running detailed description")
    update_finetuning_data(
        llm,
        json_fp="LLaVA-Instruct-150K/detail_23k.json",
        resp_type="detail_description",
        symbolic_rep_df=symbolic_rep_df,
        test_first_k=test_first_k,
        output_fp=f"generated_data/raw_detail_23k_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv",
        missing_bbox=missing_bbox,
        final_prompts=final_prompts,
    )

    # %%
    print("running complex reasoning")
    update_finetuning_data(
        llm,
        json_fp="LLaVA-Instruct-150K/complex_reasoning_77k.json",
        resp_type="complex_reasoning",
        symbolic_rep_df=symbolic_rep_df,
        test_first_k=test_first_k,
        output_fp=f"generated_data/raw_complex_reasoning_77k_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv",
        missing_bbox=missing_bbox,
        final_prompts=final_prompts,
    )

    # %%
    print("running conversation")
    update_finetuning_data(
        llm,
        json_fp="LLaVA-Instruct-150K/conversation_58k.json",
        resp_type="conversation",
        symbolic_rep_df=symbolic_rep_df,
        test_first_k=test_first_k,
        output_fp=f"generated_data/raw_conversation_58k_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv",
        missing_bbox=missing_bbox,
        final_prompts=final_prompts,
    )
