# Generate language-image instruction-following data

**Mistral assisted visual Instruction Data Generation**
(with Langchain + LLama.cpp)

>Follows Chapter 3. GPT-assisted Visual Instruction Data Generation from the [LLaVa paper](https://arxiv.org/pdf/2304.08485.pdf).

A few manually designed examples from [here](https://github.com/haotian-liu/LLaVA/tree/main/playground/data/prompts) are the only human annotations that are used as seed examples in in-context-learning to query [Mistral 7b](https://mistral.ai/news/announcing-mistral-7b/).

Three types of instruction-following data are generated this way:
1. Conversation
2. Detailed description
2. Complex reasoning

In order to encode an image into its visual features to prompt a text-only Mistral, two types of symbolic representations are used:
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

### Steps

1. download the data and gguf model

    `mistral-7b-instruct-v0.1.Q4_0.gguf`
    ```bash
    cd models

    wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_0.gguf
    ```

    `COCO 2017`
    ```bash
    cd COCO2017

    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    ```

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

pip3 install -r requirements.cpu.txt
```

3. run `main.py` interactively using `shift + return`


### Generated data
For this image [COCO_train2014_000000443336.jpg](https://knowyourdata-tfds.withgoogle.com/#dataset=coco_captions&tab=ITEM&select=default_segment.coco_captions.captions_text.value&draw=default_segment.coco_captions.objects_label.value,bbox,bbox&auto_draw=false&item=COCO_train2014_000000443336.jpg)

1. Complex reasoning
```
AI: Question:
What is the purpose of this clock?
===
Answer:
In the image, a large clock sits in the middle of a room. The clock's design suggests that it is not just an ordinary timepiece but might be used as a decorative item or as a focal point for the room. Its large size and intricate details indicate its importance to the space's overall aesthetic. It could also serve as a symbolic representation of time, organization, or precision in various contexts, such as in offices, libraries, or museums. The presence of a person interacting with the clock suggests that it may be used for display purposes only and not functional.
```

2. Conversation
```
AI: Question:
What type of clock is visible in the image?
===
Answer:
The clock visible in the image is an antique or traditional grandfather clock, with its large size and ornate design indicating that it may be considered a family heirloom or significant piece of history.
===
Question:
Is the clock functioning or just for display purposes?
===
Answer:
Based on what I can see in the photo, the clock is likely not functional, as there are no visible mechanisms, such as hands or a motor, typically found in working clocks. Instead, it appears to be purely decorative and stationary.
===
Question:
Can you tell me more about grandfather clocks? What makes them special compared to other types of clocks?
===
Answer:
Grandfather clocks are large, often ornate timepieces that typically have pendulums swinging back and forth underneath a glass cover. They originated in the 17th century as high-precision clockwork devices designed for wealthy individuals who wanted to display their wealth and status through impressive home decorations.

What sets grandfather clocks apart from other types of clocks is

```

3. Detailed description
```
AI: The image shows a large clock sitting in the middle of a room, likely an antique or decorative piece. Several people are visible near the clock, including one person who is standing directly in front of it and looking at the time with great interest, while another person stands nearby, possibly admiring its beauty.

The room itself appears to be well-maintained and spacious, with a wooden floor and walls adorned with artwork or other decorative elements. The clock is likely a focal point of the room, drawing attention from those who enter it.

```
