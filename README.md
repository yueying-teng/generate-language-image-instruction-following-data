# Generate language-image instruction-following data

**Mistral assisted visual instruction data generation**
(with Langchain + llama-cpp-python + gguf)

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
```
[
    {"role":"system", "content":"system message"},
    {"role":"user", "content":"context 000"},
    {"role":"assistant", "content":"response 000"},
    {"role":"user", "content":"context 001"},
    {"role":"assistant", "content":"response 001"},
    ...
    {"role":"user", "content":"query(Captions or Captions + Bounding boxes)"},
]
```

### Comparison with the original [LLaVA-Instruct-150K data from GPT4](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)

Overall, the generated data by [mistral-7b-instruct-v0.1.Q5_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf) is
- less structured, meaning it does not always provide an overall summary of the scene in the end.
- not able to count the number of objects. e.g. `000000154346.jpg`
- not always able to follow instructions. e.g. `000000522620.jpg`

However, with the repository's current structure, it should be relatively easy to switch to other more capable models.

[000000154346.jpg](https://knowyourdata-tfds.withgoogle.com/#dataset=coco_captions&filters=default_segment.coco_captions.image_id.value:442786&tab=ITEM&select=default_segment.coco_captions.captions_text.value&draw=default_segment.coco_captions.objects_label.value,bbox,bbox&auto_draw=false&item=COCO_train2014_000000154346.jpg)

<details>
  <summary>detail description GPT4 vs mistral-7b-instruct-v0.1.Q5_K_M.gguf</summary>

```
The image features a blue and white bus traveling down a small mountain road surrounded by large rocks. The bus appears to be a sightseeing tour bus, as several tourists are walking about and exploring the area. They are scattered along the mountainside road, with some near the bus and others further away.

In total, there are 14 people in the scene. Some are close to the rocks, likely admiring their beauty, while others are standing more towards the road. The scene conveys a sense of adventure and excitement as the tourists explore this picturesque mountain location.
```

```
The image shows a blue and white bus traveling down the street while surrounded by tourists on foot. There are several people visible in the scene, including one man holding a camera to capture memories of his trip.

The road is winding along mountainside with rocks and trees scattered around it. In addition, there are cliffs overlooking the valley below with water flowing through them. The bus stops at various points to allow passengers to take in the stunning views and explore nearby attractions such as large rocks on the side of the mountain.
```
</details>
<br/><br/>

[000000442786.jpg](https://knowyourdata-tfds.withgoogle.com/#dataset=coco_captions&filters=default_segment.coco_captions.image_id.value:000000442786&tab=ITEM&select=default_segment.coco_captions.captions_text.value&draw=default_segment.coco_captions.objects_label.value,bbox,bbox&auto_draw=false&item=COCO_train2014_000000442786.jpg)
<details>

  <summary>detail description GPT4 vs mistral-7b-instruct-v0.1.Q5_K_M.gguf</summary>

```
The scene depicts a lively plaza area with several people walking and enjoying their time. A man is standing in the plaza with his legs crossed, holding a kite in his hand. The kite has multiple sections attached to it, spread out in various directions as if ready for flight.

Numerous people are scattered throughout the plaza, walking and interacting with others. Some of these individuals are carrying handbags, and others have backpacks. The image captures the casual, social atmosphere of a bustling plaza on a nice day.
```

```
The image depicts a man flying a kite in an open plaza area filled with people walking across the pavement. The man is standing upright and holding his kite outstretched in front of him. In addition to the man, several other individuals are visible in the scene, including one person sitting on a bench nearby, another walking towards them with their mouth slightly opened, and yet another standing up with their legs crossed while holding onto their kite. The plaza area is surrounded by buildings, some of which have windows open or people inside looking out at the activity below.
```
</details>
<br/><br/>

[000000522620.jpg](https://knowyourdata-tfds.withgoogle.com/#dataset=coco_captions&filters=default_segment.coco_captions.image_id.value:442786&tab=ITEM&select=default_segment.coco_captions.captions_text.value&draw=default_segment.coco_captions.objects_label.value,bbox,bbox&auto_draw=false&item=COCO_train2014_000000522620.jpg)

<details>
  <summary>detail description GPT4 vs mistral-7b-instruct-v0.1.Q5_K_M.gguf</summary>

```
The image depicts a small bathroom with a sink, a mirror, and a bathtub. There is a woman taking a picture of herself in the bathroom mirror, capturing her upper body. On and around the sink, there are several objects, including a cup, three toothbrushes, and two bottles, likely containing toiletries.

Adjacent to the sink, a potted plant is placed, adding a touch of greenery to the bathroom. The primary color palette of the bathroom consists mainly of white, giving it a clean and classic look.
```

```
mirror: [0.371, 0.623, 0.789, 0.632
```
</details>


### Project structure
```
generate-language-image-instruction-following-data
├─ COCO2017               <-- COCO2017 annotation data will be downloaded here
├─ models                 <-- mistral 7B instruct model will be downloaded here
├─ generated_data         <-- generated raw & post processed instruction following data will be stored here
├─ coco_symbolic_representations.py   <-- run this to create symbolic representations for prompting LLMs
├─ data_utils.py
├─ instruct_158k.py                   <-- run this to generate raw instruction following data
├─ instruct_158k_with_batching.py     <-- batch prediction is not supported by langchain.llms.LlamaCpp yet
├─ model_utils.py
├─ post_process_instruct_158k.py      <-- run this to recreate LLaVA-Instruct-150K using the generated raw data
├─ pretrain_595k.py                   <-- run this to replace the instructions in the original LLaVA-CC3M-Pretrain-595K
├─ prompts                            <-- manually designed seed examples from LLaVA's official repo,
│  ├─ complex_reasoning                   which are used to prompt mistral 7B to generate instruction
│  │  ├─ ...txt                           following data in this repo
│  │  └─ system_message.txt
│  ├─ conversation
│  │  ├─ ...txt
│  │  └─ system_message.txt
│  └─ detail_description
│     ├─ ...txt
│     └─ system_message.txt
├─ requirements.cpu.txt
├─ requirements.gpu.txt
├─ run_on_test_queries.py            <-- run this to generate the three types of instruction following
├─ archive                               data for COCO image 000000443336.jpg
│  └─ eda.py
└─ test_queries                      <-- symbolic representations of COCO image 000000443336.jpg
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

pip install -r requirements.cpu.txt

# to run llama-cpp-python on NVIDIA GPUs
pip install -r requirements.gpu.txt

export CMAKE_ARGS=-DLLAMA_CUBLAS=on
export FORCE_CMAKE=1
pip install llama_cpp_python==0.2.43 --force-reinstall --upgrade --no-cache-dir
# Use --verbose for extra assurance that cuBLAS is being used in compilation.
```

3. install `Jupyter` extension first in VS Code, then run `run_on_test_queries.py` interactively using `shift + return`.

4. Run `pretrain_595k.py` to replace the original instructions in [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/chat.json) with the ones from the `replacing_instructions_brief` list [here](data_utils.py). Results will be saved as `generated_data/chat.json`, which has the same format as the original `chat.json`.

5. Recreate the three types of instruction following data, `detailed description`, `complex reasoning` and `conversation`, in `LLaVA-Instruct-150K` using Mistral instruct 7B.
    ```bash
    nohup python -u instruct_158k.py > instruct_158k.out &
    ```
    1. Run [instruct_158k.py](instruct_158k.py) to created raw instruction following data
    2. The generated raw data is saved in
        - `generated_data/raw_detail_23k.pkl`
        - `generated_data/raw_complex_reasoning_77k.pkl`
        - `generated_data/raw_conversation_58k.pkl` respectively.
    3. Then run `post_process_instruct_158k.py` to parse the generated text and recreate the LLaVA training data, which share the same format as those in [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K), and will be saved in
        - `generated_data/complex_reasoning_77k.json`
        - `generated_data/conversation_58k.json`
        - `generated_data/detail_23k.json`

### Generated data
For this image [COCO_train2014_000000443336.jpg](https://knowyourdata-tfds.withgoogle.com/#dataset=coco_captions&tab=ITEM&select=default_segment.coco_captions.captions_text.value&draw=default_segment.coco_captions.objects_label.value,bbox,bbox&auto_draw=false&item=COCO_train2014_000000443336.jpg), results from running [run_on_test_queries.py](run_on_test_queries.py).

1. Complex reasoning
```
AI: Question:
What is the purpose of this clock?
===
Answer:
In the image, a large clock sits in the middle of a room. The clock's design and size suggest that it might be an antique or a significant piece of artwork, rather than just a functional timepiece. It could be on display as part of a museum exhibit or collection, showcasing its craftsmanship and historical significance. Alternatively, the clock could be used in interior decoration to add a unique and striking element to the room's aesthetic. However, without more context, it is impossible to determine the exact purpose of this clock with certainty.
```

2. Conversation
```
AI: Question:
What kind of clock is shown in the image?
===
Answer:
The clock shown in the image appears to be a large and decorative wooden grandfather clock, with glass elements that reveal its inner workings. It has an intricate design on its face, suggesting it might have historical or artistic significance.
===
Question:
Is this clock functional or purely for display purposes?
===
Answer:
Based on the image provided, it is difficult to determine whether the grandfather clock in the middle of the room is entirely functional or simply a decorative piece. However, given its size and detailed design, it seems more likely that the clock would be displayed rather than used as an actual timepiece for practical purposes.
===
Question:
What materials make up this clock?
===
Answer:
The grandfather clock in the image appears to be made primarily of wood, with glass elements showcasing its inner workings and revealing some of the design details on the face. Wood is commonly used in making clocks due to its durability, versatility, and aesthetic appeal. The specific type of wood or finish applied can influence the overall appearance and character of the clock.
===
```

3. Detailed description
```
AI: The image is a close-up of a large clock sitting on the floor in the middle of a room. There are three people visible in the scene - one man standing directly in front of the clock, another person standing to its right, and a third individual standing farther away towards the left side of the frame.

The clock appears intricately designed with a large wooden face surrounded by decorative elements made from glass and metal. The hands of the clock are visible at 12:30, indicating that it is almost time for something important to happen or that an event has already started.
```
