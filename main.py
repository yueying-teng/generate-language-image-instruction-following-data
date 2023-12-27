# %%
import os
import glob

from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

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

# %%
def read_file(fp):
    with open(fp, "r") as file:
        result = file.read()

    return result

# %%
def get_prompt(prompt_dir):
    cap_fps = glob.glob(f"./prompts/{prompt_dir}/*_caps.txt")
    conv_fps = glob.glob(f"./prompts/{prompt_dir}/*_conv.txt")

    sys_msg_path = f"./prompts/{prompt_dir}/system_message.txt"
    sys_msg = read_file(sys_msg_path)

    examples = []
    for cap_fp, conv_fp in zip(sorted(cap_fps), sorted(conv_fps)):
        cap = read_file(cap_fp)
        conv = read_file(conv_fp)
        examples.append({"input": cap, "output": conv})

    # This is a prompt template used to format each individual example.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_msg),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    return final_prompt


# %%
final_prompts = {}
for prompt_dir in sorted(os.listdir("prompts")):
    # detail_description, complex_reasoning, conversation
    if prompt_dir != ".DS_Store":
        final_prompts[prompt_dir] = get_prompt(prompt_dir)

llm_chains = {}
for resp_type, prompt in final_prompts.items():
    # llm_chain = LLMChain(prompt=prompt, llm=llm) # Legacy code
    llm_chains[resp_type] = prompt | llm  # LCEL

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
# for fp in description_test_fps:
#     cap = read_file(fp)
#     for resp_type, llm_chain in llm_chains.items():
#         print(f"{resp_type} .....................")
#         response = llm_chain.invoke({"input": cap})

#         print(response)

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
