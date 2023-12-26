import streamlit as st
from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


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
        max_tokens=512,
        seed=2023,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        # n_gpu_layers=1,
        # n_batch=512,
        # n_ctx=4096,
        stop=["[INST]"],
        verbose=False,
        )


examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

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

# We create a prompt from the template so we can use it with Langchain
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# We create an llm chain with our LLM and prompt
# llm_chain = LLMChain(prompt=prompt, llm=llm) # Legacy
llm_chain = final_prompt | llm  # LCEL


st.set_page_config(page_title="instruction following data Generator", page_icon="#️⃣", layout="wide", )
st.markdown(f"""
            <style>
            .stApp {{background-image: url("https://images.unsplash.com/photo-1612538498456-e861df91d4d0?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1974&q=80");
                     background-attachment: fixed;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)

st.title("Enter the instruction")
instruction = st.text_input("enter the instruction here", placeholder="6+8",)
if instruction:
    response = response = llm_chain.invoke({"input": instruction})
    st.write(response)
