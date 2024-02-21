# Generate language-image instruction-following data

**Mistral assisted visual instruction data generation**
(with llama-cpp-python + gguf)

>Follows Chapter 3. GPT-assisted Visual Instruction Data Generation from the [LLaVa paper](https://arxiv.org/pdf/2304.08485.pdf).

A few manually designed examples from [here](https://github.com/haotian-liu/LLaVA/tree/main/playground/data/prompts) are the only human annotations that are used as seed examples in in-context-learning to query [Mistral 7b instruct v0.1](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf).

Three types of instruction-following data are generated in this way:
1. Conversation: [conversation_58k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/conversation_58k.json)
2. Detailed description: [detail_23k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/detail_23k.json)
3. Complex reasoning: [complex_reasoning_77k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/complex_reasoning_77k.json)

In order to encode an image into its visual features to prompt a text-only Mistral, two types of symbolic representations from `COCO annotations_trainval2017` are used:
1. Captions
2. Bounding boxes


### Overall prompt structure for each image

According to `Mistral-7B-Instruct-v0.1`'s `chat_template` specified [here](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer_config.json#L32), system message is not supported.
One remedy as suggested in Mistral's [documentation](https://web.archive.org/web/20231030013339/https://docs.mistral.ai/usage/guardrailing/#appendix) is to prepend the system message to the first user messagem, leading to an overall prompt sructure like below:
```
[
    {"role":"user", "content":"system message" + "context 000"},
    {"role":"assistant", "content":"response 000"},
    {"role":"user", "content":"context 001"},
    {"role":"assistant", "content":"response 001"},
    ...
    {"role":"user", "content":"context query (Captions or Captions + Bounding boxes)"},
]
```


### Project structure
```
generate-language-image-instruction-following-data
├─ COCO2017               <-- COCO2017 annotation data will be downloaded here
├─ models                 <-- mistral 7B instruct model will be downloaded here
├─ generated_data         <-- generated raw & post processed instruction following data will be stored here
├─ symbolic_rep_data      <-- symbolic representations created from COCO2017 annotation data will be saved here
├─ src
│  ├─ coco_symbolic_representations.py   <-- run this to create symbolic representations for prompting Mistral
│  ├─ data_utils.py
│  ├─ instruct_158k.py                   <-- run this to generate all three types of raw instruction following data
│  ├─ model_utils.py
│  ├─ post_process_instruct_158k.py      <-- run this to recreate LLaVA-Instruct-150K using the raw instruction following data
│  ├─ pretrain_595k.py                   <-- run this to replace the instructions in the original LLaVA-CC3M-Pretrain-595K
|  └─ run_on_test_queries.py
├─ prompts                               <-- manually designed seed examples from LLaVA's official repo
│  ├─ complex_reasoning
│  │  ├─ ...txt
│  │  └─ system_message.txt
│  ├─ conversation
│  │  ├─ ...txt
│  │  └─ system_message.txt
│  └─ detail_description
│     ├─ ...txt
│     └─ system_message.txt
├─ requirements.txt
├─ archive                               data for COCO image 000000443336.jpg
│  └─ eda.py
└─ test_queries                      <-- symbolic representations of one example COCO image 000000443336.jpg
   ├─ complex_reasoning
   │  └─ 000_caps.txt
   ├─ conversation
   │  └─ 000_caps.txt
   └─ detail_description
      └─ 000_caps.txt
```


### Steps

1. download the data and gguf model

    `mistral-7b-instruct-v0.1.Q5_K_M.gguf`
    ```bash
    cd models

    wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf

    wget https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf
    ```
    GCP VM spec necessary to run `mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf`: g2-standard-48 (4 NVIDIA L4 GPU).

    `COCO 2017`
    ```bash
    cd COCO2017

    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    ```

    **make sure `git lfs` is install first**

    [`LLaVA-Instruct-150K`](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
    ```bash
    git lfs install
    git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
    ```

    [`LLaVA-CC3M-Pretrain-595K`](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K)
    ```bash
    git lfs install
    git clone https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K
    ```


2. create conda environment and install dependencies
```bash
conda create -n mistral python=3.10.0

conda activate mistral

pip install -r requirements.txt

# to run llama-cpp-python on NVIDIA GPUs
export CMAKE_ARGS=-DLLAMA_CUBLAS=on
export FORCE_CMAKE=1
pip install llama_cpp_python==0.2.43 --force-reinstall --upgrade --no-cache-dir
# Use --verbose for extra assurance that cuBLAS is being used in compilation.
```

3. Install `Jupyter` extension first in VS Code, then run `src/run_on_test_queries.py` interactively using `shift + return`.

4. Run `pretrain_595k.py` to replace the original instructions in [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/chat.json) with the ones from the `replacing_instructions_brief` list [here](src/data_utils.py). Results will be saved as `generated_data/chat.json`, which has the same format as the original `chat.json`.

5. Recreate the three types of instruction following data, `detailed description`, `complex reasoning` and `conversation`, in `LLaVA-Instruct-150K` using Mistral instruct 7B.
    ```bash
    nohup python -u src/instruct_158k.py > instruct_158k.out &
    ```
    1. Run [instruct_158k.py](src/instruct_158k.py) to created raw instruction following data
    2. The generated raw data is saved in
        - `generated_data/raw_detail_23k_{%Y_%m_%d_%H_%M_%S}.csv`
        - `generated_data/raw_complex_reasoning_77k_{%Y_%m_%d_%H_%M_%S}.csv`
        - `generated_data/raw_conversation_58k_{%Y_%m_%d_%H_%M_%S}.csv` respectively.
    3. Then run `src/post_process_instruct_158k.py` to parse the generated text and recreate the LLaVA training data, which share the same format as those in [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K). They will be saved in
        - `generated_data/complex_reasoning_77k.json`
        - `generated_data/conversation_58k.json`
        - `generated_data/detail_23k.json`


### Example generated data using `Mistral-7B-Instruct-v0.1`

For this image [COCO_train2014_000000443336.jpg](https://knowyourdata-tfds.withgoogle.com/#dataset=coco_captions&tab=ITEM&select=default_segment.coco_captions.captions_text.value&draw=default_segment.coco_captions.objects_label.value,bbox,bbox&auto_draw=false&item=COCO_train2014_000000443336.jpg), results from running [run_on_test_queries.py](src/run_on_test_queries.py).

1. Complex reasoning
```
Question:
What is unusual about this photo?
===
Answer:
In this photo, there is a large clock sitting on the floor and displayed in a room. The clock's design is quite unique, featuring wood and glass construction. The clock's large size and mechanical components suggest it is a grandfather clock. The clock's placement at the center of the room draws attention to it and might be used as a focal point or decorative element to enhance the room's ambiance. The presence of people near the clock is unusual because it's not common to see people interacting directly with a clock on display, especially one as large and intricate as this one.
```

2. Conversation
```
Question:
What is the clock doing in the room? Is it functional or just for display?
===
Answer:
The clock in the room appears to be functional, as it is large and has a grandfather clock design. However, without more context or information on its surroundings or other objects in the room, it is difficult to determine its exact purpose or intended use.

In contrast to a clock on display in a museum, which is typically a historical or antique piece meant to be admired for its beauty, craftsmanship, or historical significance, the clock in the room seems to be more practical or functional in nature.

The clock design suggests that it may be used to indicate time or elapsed time for various purposes, such as work schedules, personal timekeeping, or even coordinating activities within a household or organization.
However, without additional details, it is not possible to determine its specific function or intended purpose with certainty.
```

3. Detailed description
```
The image shows a large clock sitting in the middle of a room. The clock is likely a grandfather clock, with its intricate wooden and glass design prominent in the photograph. The clock appears to be on display in a museum or historical building, surrounded by empty space and possibly other antiques or artifacts.

There is also one person visible in the photograph, standing to the right side of the clock and facing it. The person appears to be admiring or examining the clock closely. Another person is standing nearby but out of the frame, likely also looking at the clock or taking photographs.
```
