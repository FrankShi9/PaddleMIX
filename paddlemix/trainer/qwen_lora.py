from paddlenlp.transformers import Qwen2Tokenizer
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.processors.qwen2_vl_processing import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    process_vision_info,
)
import loralib


if __name__ == "__main__":

    MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, dtype="bfloat16")

    image_processor = Qwen2VLImageProcessor()
    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_NAME)
    processor = Qwen2VLProcessor(image_processor, tokenizer)

    # min_pixels = 256*28*28 # 200704
    # max_pixels = 1280*28*28 # 1003520
    # processor = Qwen2VLProcessor(image_processor, tokenizer, min_pixels=min_pixels, max_pixels=max_pixels)


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "paddlemix/demo_images/examples_image1.jpg"},
                {"type": "image", "image": "paddlemix/demo_images/examples_image2.jpg"},
                {"type": "text", "text": "Identify the similarities between these images."},
            ],
        }
    ]

    # Preparation for inference
    image_inputs, video_inputs = process_vision_info(messages)

    question = "Identify the similarities between these images."
    image_pad_tokens = "<|vision_start|><|image_pad|><|vision_end|>" * len(image_inputs)
    text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_pad_tokens}{question}<|im_end|>\n<|im_start|>assistant\n"

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pd",
    )

    # lora
    model.apply_lora_(loralib.Linear(in_features=model.config["hidden_size"], out_features=model.config["hidden_size"]))
    model.train()
    loss = paddle.nn.MSELoss()

    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())

    for param in model.parameters():
        if param.name == 'lora_weight':
            param.stop_gradient = False
            param.register_hook(lambda g: print('Grad: ', g))
            get_param_info(param)

    epochs = 10
    save_interval = 1

    # train loop
    for i in range(epochs):
        outputs = model(**inputs)

        loss = outputs.logits.mean().backward()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch: ", i)
        print("loss: ", loss)

        for param in model.parameters():
            if param.name == 'lora_weight':
                print(get_lora_info())

    
        if i+1 % save_interval == 0:
            model.save_pretrained(f'checkpoints/{i+1}_epoch')
            print(f'Saved checkpoint at epoch {i+1}')

        
    print("Training finished.")


    

