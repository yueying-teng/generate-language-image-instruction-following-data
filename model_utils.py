import os
import glob

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


def read_file(fp):
    with open(fp, "r") as file:
        result = file.read()

    return result


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


def get_llm_chains(llm):
    final_prompts = {}
    for prompt_dir in sorted(os.listdir("prompts")):
        # detail_description, complex_reasoning, conversation
        if prompt_dir != ".DS_Store":
            final_prompts[prompt_dir] = get_prompt(prompt_dir)

    llm_chains = {}
    for resp_type, prompt in final_prompts.items():
        # llm_chain = LLMChain(prompt=prompt, llm=llm) # Legacy code
        llm_chains[resp_type] = prompt | llm  # LCEL

    return llm_chains
