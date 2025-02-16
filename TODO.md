## Inference improvement

ggml models are not thread safe
https://github.com/marella/ctransformers/issues/38#issuecomment-1613627309

`llama_cpp.Llama.create_chat_completion` does not support batch inference

with one GPU
- due to GIL, multi-threading has no effect
    - if one thread has access to the model, then no other thread can access the model
    until the first thread finishes dealing with the model
    - essentially, there is no performance gain from multi-threading
    - unless inference also waits on I/O operations, then multi-threading might be useful
- for multi-processing,
    - spawn python processes that can inherit the current Python interpreter process’ resources
    (bypassing the GIL issue, using the multiprocessing module).
    - each process will need to load the model to memory, with one GPU, this will consume too much memory

with multiple GPUs
- multi-processing will be more effective
    - but without the support of batch inference, each GPU won't be fully utilized

the only option is to explore other libraries and model formats
