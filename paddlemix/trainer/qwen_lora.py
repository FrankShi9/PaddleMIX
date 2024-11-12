from paddlenlp.transformers import Qwen2Tokenizer
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.processors.qwen2_vl_processing import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    process_vision_info,
)

import loralib


def get_param_info(param):
    """
    打印参数信息。
    
    Args:
        param (object): 待打印信息的参数对象。
    
    Returns:
        None
    
    """
    print("param name: ", param.name)
    print("param type: ", type(param))
    print("param dtype: ", param.dtype)
    print("param shape: ", param.shape)
    print("param data: ", param.data)
    print("param stop_gradient: ", param.stop_gradient)
    print("param is_leaf: ", param.is_leaf)
    print("param requires_grad: ", param.requires_grad)
    print("param trainable: ", param.trainable)
    print("param grad: ", param.grad)
    print("param grad_fn: ", param.grad_fn)
    print("param grad_req: ", param.grad_req)

def get_lora_info(lora_layer):
    print("lora_layer name: ", lora_layer.name)
    print("lora_layer type: ", type(lora_layer))
    print("lora_layer dtype: ", lora_layer.dtype)
    print("lora_layer shape: ", lora_layer.shape)
    print("lora_layer data: ", lora_layer.data)
    print("lora_layer stop_gradient: ", lora_layer.stop_gradient)
    print("lora_layer is_leaf: ", lora_layer.is_leaf)
    print("lora_layer requires_grad: ", lora_layer.requires_grad)
    print("lora_layer trainable: ", lora_layer.trainable)
    print("lora_layer grad: ", lora_layer.grad)
    print("lora_layer grad_fn: ", lora_layer.grad_fn)
    print("lora_layer grad_req: ", lora_layer.grad_req)

    print("lora weight: ", model.get_lora_weight())
    print("lora bias: ", model.get_lora_bias())
    print("lora weight grad: ", model.get_lora_weight_grad())
    print("lora bias grad: ", model.get_lora_bias_grad())
    print("lora weight norm: ", model.get_lora_weight_norm())
    print("lora bias norm: ", model.get_lora_bias_norm())
    print("lora weight rms: ", model.get_lora_weight_rms())
    print("lora bias rms: ", model.get_lora_bias_rms())
    print("lora weight decay: ", model.get_lora_weight_decay())
    print("lora bias decay: ", model.get_lora_bias_decay())
    print("lora weight decay rate: ", model.get_lora_weight_decay_rate())
    print("lora bias decay rate: ", model.get_lora_bias_decay_rate())


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

    # lora 训练
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


    

