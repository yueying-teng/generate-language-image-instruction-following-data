# %%
import json

import pandas as pd
from langchain.llms import LlamaCpp

from pretrain_595k import replace_instruction, patterns, save_to_json
from model_utils import get_llm_chains

#%%
symbolic_rep_df = pd.read_pickle("symbolic_representation_coco_trainval_2017.pkl")
symbolic_rep_df.head()

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

        list_data_dict[i]["conversations"][1]["value"] = response[5:]

    save_to_json(list_data_dict, output_fp)

# %%
replacing_instructions = [
    "testing 01",
    "testing 02",
    "testing 03",
]

update_detail_23k(
    detail_23k_fp="../LLaVA-Instruct-150K/detail_23k.json",
    replacing_instructions=replacing_instructions,
    symbolic_rep_df=symbolic_rep_df,
    llm_chains=llm_chains,
)


# %%
def update_complex_77k(
    complex_77k_fp,
    symbolic_rep_df,
    llm_chains,
    test_first_k=2,  # update the entire complex_77k data using float("-inf")
    output_fp="updated_complex_77k.json",
):
    list_data_dict = json.load(open(complex_77k_fp, "r"))

    for i in range(min(test_first_k, len(list_data_dict))):
        image = list_data_dict[i]["image"]
        symbolic_rep = symbolic_rep_df[symbolic_rep_df["image"] == image].iloc[0]

        human_input = symbolic_rep["caption"] + "\n\n" + symbolic_rep["bbox"]
        response = llm_chains["complex_reasoning"].invoke({"input": human_input})

        qa = response.split("Question:")[-1].split("===")
        question = qa[0].rstrip("\n").strip("\n")
        answer = qa[-1].split("Answer:")[-1].strip("\n")

        _ = replace_instruction([list_data_dict[i]], [question], patterns)

        list_data_dict[i]["conversations"][1]["value"] = answer

    save_to_json(list_data_dict, output_fp)

# %%
update_complex_77k(
    complex_77k_fp="../LLaVA-Instruct-150K/complex_reasoning_77k.json",
    symbolic_rep_df=symbolic_rep_df,
    llm_chains=llm_chains,
    test_first_k=2,  # update the entire complex_77k data using float("-inf")
    output_fp="updated_complex_77k.json",
)
