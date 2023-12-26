# Visual instruction following data generation

## Mistral w/ Langchain + LLama.cpp

[Mistral 7b](https://mistral.ai/news/announcing-mistral-7b/)


### Steps

1. download the gguf model `mistral-7b-instruct-v0.1.Q4_0.gguf`
```bash
cd models

wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_0.gguf
```

2. create conda environment and install dependencies
```bash
conda create -n mistral python=3.10.0

conda activate mistral

pip3 install -r requirements.txt
```

3. run `main.py` interactively using `shift + return`


### Others

[Explore COCO dataset](https://knowyourdata-tfds.withgoogle.com/#dataset=coco_captions&tab=ITEM&select=default_segment.coco_captions.captions_text.value&draw=default_segment.coco_captions.objects_label.value,bbox,bbox&auto_draw=false&item=COCO_train2014_000000442786.jpg)

- COCO_train2014_000000442786.jpg
- COCO_train2014_000000443336.jpg
