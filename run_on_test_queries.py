# %%
import os
import glob

from langchain.llms import LlamaCpp

from model_utils import get_llm_chains, get_prompt, read_file


# initialize LlamaCpp LLM model
# n_gpu_layers, n_batch, and n_ctx are for GPU support.
# When not set, CPU will be used.
# set 1 for Mac m2, and higher numbers based on your GPU support
llm = LlamaCpp(
        model_path="./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
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

# %%
final_prompts = {}
for prompt_dir in sorted(os.listdir("prompts")):
    # detail_description, complex_reasoning, conversation
    if prompt_dir != ".DS_Store":
        final_prompts[prompt_dir] = get_prompt(os.path.join("prompts", prompt_dir))

llm_chains = get_llm_chains(llm)
# %%
# inspect the final prompt
for resp_type, prompt in final_prompts.items():
    print(resp_type)
    test_fps = sorted(glob.glob(f"./test_queries/{resp_type}/*_caps.txt"))
    print(test_fps)
    for fp in test_fps:
        cap = read_file(fp)
        p = prompt.invoke({"input": cap})
        # print(p.to_messages())
        print(p.to_string())


# %%
for resp_type, prompt in final_prompts.items():
    print(f"{resp_type} .....................")
    test_fps = sorted(glob.glob(f"./test_queries/{resp_type}/*_caps.txt"))
    print(test_fps)

    for fp in test_fps:
        cap = read_file(fp)
        response = llm_chains[resp_type].invoke({"input": cap})

        print(response)

# %%
