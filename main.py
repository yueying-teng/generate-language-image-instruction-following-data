# %%
import glob

from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# %%
model_path = "./models/mistral-7b-instruct-v0.1.Q4_0.gguf"

cap_fps = glob.glob("./prompts/detail_description/*_caps.txt")
conv_fps = glob.glob("./prompts/detail_description/*_conv.txt")

# %%
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
        max_tokens=128,
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

sys_msg_path = "./prompts/detail_description/system_message.txt"
sys_msg = read_file(sys_msg_path)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sys_msg),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# llm_chain = LLMChain(prompt=prompt, llm=llm) # Legacy code
llm_chain = final_prompt | llm  # LCEL

# %%
description_test_fps = glob.glob("./test_queries/detail_description/*_caps.txt")
for fp in description_test_fps:
    cap = read_file(fp)
    # inspect the final prompt
    prompt = final_prompt.invoke({"input": cap})
    # print(prompt.to_messages())
    print(prompt.to_string())


# %%
for fp in description_test_fps:
    cap = read_file(fp)
    response = llm_chain.invoke({"input": cap})

    print(response)
