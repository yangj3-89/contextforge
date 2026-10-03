# Project Wren

Project Wren is Tessellate AI's on-device inference runtime for field technicians. It runs a distilled 1.3-billion-parameter language model directly on tablets, so technicians can query repair manuals without network access.

## Team

Samantha Reyes leads Project Wren from the Berlin office. Marta Kowalski joined the team in January 2026 to work on the quantization pipeline.

## How it works

Models are exported to ONNX and executed with ONNX Runtime on Android tablets and with Core ML on iPads. Weights are quantized to int8 with per-channel scales, which cut the model size from 2.6 GB to 1.4 GB.

## Performance

On the reference mid-range Android tablet, Wren generates the first token in 180 ms at p95 and sustains 14 tokens per second. Battery drain during a 10-minute session stays under 4 percent.

## Limitations

Wren does not support languages other than English and German yet. Context length is capped at 2,048 tokens to stay within device memory.
