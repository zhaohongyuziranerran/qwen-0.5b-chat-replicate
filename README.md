# Qwen 0.5B Chat - Replicate Model

A fast and efficient Chinese language model for conversation, deployed on [Replicate](https://replicate.com).

## Features

- 0.5B parameters, runs on single GPU
- Chinese language optimized
- Fast inference speed
- Low memory footprint

## Use Cases

- Chatbots
- Content generation
- Question answering
- Text completion

## Quick Start

```python
import replicate

# Run the model
output = replicate.run(
    "zhaohongyu/qwen-0.5b-chat-replicate:latest",
    input={
        "prompt": "Hello, how are you?",
        "max_tokens": 512,
        "temperature": 0.7
    }
)

print(output)
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prompt | string | - | Your input text |
| max_tokens | integer | 512 | Maximum tokens to generate |
| temperature | float | 0.7 | Sampling temperature (0-2) |

## Hardware Requirements

- GPU: T4
- Memory: 8GB

## License

Apache 2.0
