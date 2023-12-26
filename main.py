# %%
import glob

# import streamlit as st
from langchain.llms import LlamaCpp
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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
        max_tokens=64,
        seed=2023,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        # n_gpu_layers=1,
        # n_batch=512,
        n_ctx=4096,
        # stop=["[INST]"],
        verbose=False,
        )

# %%
def read_file(fp):
    with open(fp, 'r') as file:
        result = file.read()

    return result

sys_msg_path = "./prompts/detail_description/system_message.txt"
sys_msg = read_file(sys_msg_path)

system_message_prompt = SystemMessagePromptTemplate.from_template(sys_msg)

few_shots = []
for cap_fp, conv_fp in zip(sorted(cap_fps), sorted(conv_fps)):
    cap = read_file(cap_fp)
    conv = read_file(conv_fp)
    human_message_prompt = HumanMessagePromptTemplate.from_template(cap)
    ai_message_prompt = AIMessagePromptTemplate.from_template(conv)
    few_shots.append(human_message_prompt)
    few_shots.append(ai_message_prompt)

query_template = """{cap}"""
query_message_prompt = HumanMessagePromptTemplate.from_template(query_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt] + few_shots + [query_message_prompt]
)
# We create an llm chain with our LLM and prompt
# llm_chain = LLMChain(prompt=prompt, llm=llm) # Legacy
llm_chain = chat_prompt | llm  # LCEL

# %%
description_test_fps = glob.glob("./test_queries/detail_description/*_caps.txt")
for fp in description_test_fps:
    cap = read_file(fp)
    response = llm_chain.invoke({"cap": cap})

    print(response)


# %%
