# %%
import glob

from llama_cpp import Llama

from model_utils import get_all_prompts, read_file


llm = Llama(
    model_path="./models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
    n_batch=4096, # Number of tokens to process in parallel. Should be a number between 1 and n_ctx
    n_ctx=4096,
    n_gpu_layers=33,
    chat_format="llama-2",
    verbose=False,
)

# %%
final_prompts = get_all_prompts("prompts")

for resp_type, prompts in final_prompts.items():
    print(f"{resp_type} .....................")
    test_fps = sorted(glob.glob(f"./test_queries/{resp_type}/*_caps.txt"))
    print(test_fps)

    for fp in test_fps:
        human_input = read_file(fp)

        llm.reset()
        resp = llm.create_chat_completion(
                messages = prompts + [
                    {
                        "role": "user",
                        "content": human_input,
                    },
                ],
                temperature=0.7,
                repeat_penalty=1.2,
                top_p=1.,
                top_k=40,
                max_tokens=256,
                seed=2023,
            )

    print(resp["choices"][0]["message"]["content"])
