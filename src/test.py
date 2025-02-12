import os
import csv
import json
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
from datetime import datetime
from llama_cpp import Llama
from model_utils import SAMPLING_CONFIG, get_all_prompts

# Set the distributed environment
def setup(rank, world_size):
    """Initialize distributed processing."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Destroy the distributed process group."""
    dist.destroy_process_group()

# Remove images with missing bounding boxes
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

# Batch processing for Llama predictions
def complete(llm, human_inputs, prompts, sampling_config):
    llm.reset()

    batch_prompts = [
        prompts + [{"role": "user", "content": human_input}]
        for human_input in human_inputs
    ]

    responses = []
    for prompt_batch in batch_prompts:
        resp = llm.create_chat_completion(
            messages=prompt_batch,
            temperature=sampling_config["temperature"],
            repeat_penalty=sampling_config["repeat_penalty"],
            top_p=sampling_config["top_p"],
            top_k=sampling_config["top_k"],
            max_tokens=sampling_config["max_tokens"],
            seed=2023,
        )
        responses.append(resp["choices"][0]["message"]["content"])

    return responses

# Batch processing using DistributedDataParallel
def predict_and_save(
    rank, world_size, llm, list_data_dict, symbolic_rep_df, final_prompts, test_first_k,
    resp_type, starting_img, output_fp, batch_size
):
    size = min(len(list_data_dict), test_first_k)
    images = [list_data_dict[i]["image"] for i in range(size)]

    # Split work among GPUs
    chunk_size = len(images) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank != world_size - 1 else len(images)

    local_images = images[start_idx:end_idx]

    with open(f"{output_fp}_rank_{rank}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "response"])

        for i in range(0, len(local_images), batch_size):
            batch_images = local_images[i : i + batch_size]
            human_inputs = [
                symbolic_rep_df[symbolic_rep_df["image"] == img].iloc[0]["human_input"]
                for img in batch_images
            ]

            responses = complete(
                llm,
                human_inputs,
                final_prompts[resp_type],
                SAMPLING_CONFIG[resp_type],
            )

            for img, res in zip(batch_images, responses):
                writer.writerow([img, res])

    # Sync all processes
    dist.barrier()

# Update fine-tuning data
def update_finetuning_data(
    rank, world_size, llm, json_fp, resp_type, symbolic_rep_df, test_first_k,
    output_fp, missing_bbox, final_prompts, batch_size
):
    assert resp_type in ["complex_reasoning", "detail_description", "conversation"], \
        "resp_type must be in [complex_reasoning, detail_description, conversation]"

    with open(json_fp) as f:
        list_data_dict = json.load(f)

    if resp_type == "complex_reasoning":
        list_data_dict = remove_image_with_missing_bbox(list_data_dict, missing_bbox)

    if resp_type in ["complex_reasoning", "detail_description"]:
        symbolic_rep_df["human_input"] = symbolic_rep_df.apply(
            lambda row: str(row["caption"]) + "\n\n" + str(row["bbox"]),
            axis=1,
        )
    else:
        symbolic_rep_df["human_input"] = symbolic_rep_df["caption"]

    predict_and_save(
        rank, world_size, llm, list_data_dict, symbolic_rep_df, final_prompts,
        test_first_k, resp_type, -1, output_fp, batch_size
    )

# Main function for distributed execution
def main(rank, world_size):
    setup(rank, world_size)

    symbolic_rep_df = pd.read_pickle("symbolic_rep_data/symbolic_representation_instruct_150k.pkl")
    missing_bbox = pd.read_pickle("symbolic_rep_data/instruct_150k_missing_bbox.pkl")["image"]

    llm = Llama(
        model_path="./models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf",
        n_batch=6200,
        n_ctx=6200,
        n_gpu_layers=33,
        chat_format="mistral-instruct",
        verbose=False,
    ).to(rank)  # Move model to the assigned GPU

    # Wrap in DistributedDataParallel
    llm = torch.nn.parallel.DistributedDataParallel(llm, device_ids=[rank])

    final_prompts = get_all_prompts("prompts")
    test_first_k = sys.maxsize
    batch_size = 8

    tasks = [
        {
            "json_fp": "LLaVA-Instruct-150K/detail_23k.json",
            "resp_type": "detail_description",
            "output_fp": f"generated_data/raw_detail_23k_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        },
        {
            "json_fp": "LLaVA-Instruct-150K/complex_reasoning_77k.json",
            "resp_type": "complex_reasoning",
            "output_fp": f"generated_data/raw_complex_reasoning_77k_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        },
        {
            "json_fp": "LLaVA-Instruct-150K/conversation_58k.json",
            "resp_type": "conversation",
            "output_fp": f"generated_data/raw_conversation_58k_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        },
    ]

    for task in tasks:
        update_finetuning_data(
            rank, world_size, llm, task["json_fp"], task["resp_type"],
            symbolic_rep_df, test_first_k, task["output_fp"],
            missing_bbox, final_prompts, batch_size
        )

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
