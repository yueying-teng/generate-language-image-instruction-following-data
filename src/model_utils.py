import os
import glob


SAMPLING_CONFIG = {
    "complex_reasoning": {
        "temperature": 1.7,
        "repeat_penalty": 1.2,
        "top_p": 0.6,
        "top_k": 40,
        "max_tokens": 340,
    },
    "detail_description": {
        "temperature": 1.5,
        "repeat_penalty": 1.2,
        "top_p": 0.59,
        "top_k": 50,
        "max_tokens": 200,
    },
    "conversation": {
        "temperature": 1.7,
        "repeat_penalty": 1.25,
        "top_p": 0.5,
        "top_k": 40,
        "max_tokens": 710,
    },
}


def read_file(fp):
    with open(fp, "r") as file:
        result = file.read()

    return result


def get_prompt(prompt_dir):
    cap_fps = glob.glob(f"{prompt_dir}/*_caps.txt")
    conv_fps = glob.glob(f"{prompt_dir}/*_conv.txt")

    sys_msg_path = f"{prompt_dir}/system_message.txt"
    sys_msg = read_file(sys_msg_path)

    prompts = [{"role": "system", "content": sys_msg}]
    for cap_fp, conv_fp in zip(sorted(cap_fps), sorted(conv_fps)):
        cap = read_file(cap_fp)
        conv = read_file(conv_fp)
        prompts.append({"role": "user", "content": cap})
        prompts.append({"role": "assistant", "content": conv})

    return prompts


def get_all_prompts(prompt_dir):
    prompts = {}
    for dir_name in sorted(os.listdir(prompt_dir)):
        # detail_description, complex_reasoning, conversation
        if dir_name != ".DS_Store":
            prompts[dir_name] = get_prompt(os.path.join(prompt_dir, dir_name))

    return prompts
