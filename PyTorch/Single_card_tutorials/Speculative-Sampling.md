<div align="center">
    <h2>
    <strong>IMPROVE TEXT GENERATION SPEED WITH ASSISTED DECODING FOR ANY LLM</strong>
    </h2>
</div>

<div align="center">
    <h4>SPECULATIVE SAMPLING or ASSISTED DECODING can be used as a text generation method to significantly improve text-generation throughput without compromising quality of generation for any LLM.</h4>
    
</div>

<div align="justify">

As generative AI models continue to grow in complexity and size, the resources required for inference have increased significantly. This surge in computational demand not only drives up the cost per text generation but also leads to higher power consumption. Optimizing inference is crucial for reducing latency, infrastructure costs, and energy usage, ultimately enhancing both efficiency and the user experience in AI-powered text generation.

**Enhancing AI Efficiency with Assisted Decoding**

One of the most effective methods for accelerating text generation is assisted decoding. At Intel, we have adapted and optimized this technique for Intel Gaudi processors, which offer performance comparable to Nvidia H100 GPUs at a price point like Nvidia A100 80GB GPUs. Our advancements in assisted decoding are now integrated into Optimum Habana, an extension of the Hugging Face ecosystem, enabling seamless AI workflow optimization for Gaudi processors.

**What is Speculative Sampling in Text Generation?**

Speculative sampling is a powerful technique that speeds up text generation while maintaining the quality of sampling. The approach involves using a draft or assistant model to generate a batch of K tokens, which are then validated against the target model. If the draft or assistant model's output is accepted, it is used as final output; otherwise, the target model generates the next token. This iterative process continues, allowing faster inference while maintaining accuracy. Thus, this method of speculatory output generation does not compromise on quality of generated text. Its worst-case speed is 1x (that of vanilla text generation for the particular LLM).

By leveraging speculative sampling, 2x speedups in large transformer-based models can be expected. This method ensures that the output distribution remains consistent with autoregressive sampling, guaranteeing high-quality text generation results.

**How does it work?**

**Optimization Challenges and KV Caching**

A key challenge in implementing speculative sampling is managing the Key-Value (KV) cache for both the draft and target models. Since these models differ in size and structure, it is crucial to apply separate optimization strategies to maximize efficiency.

For this optimization, we assume a quantized model setup and leverage KV caching alongside speculative sampling. Each model maintains its own KV cache, ensuring efficient memory usage and minimal computational overhead. The draft model generates K tokens before validation against the target model, and this cycle repeats to maintain the balance between speed and accuracy.

Research has shown that speculative sampling recovers the target distribution, ensuring the same sampling quality as autoregressive generation. However, its benefits are more pronounced when the draft model is significantly smaller than the target model and has a high acceptance rate. If these conditions are not met, the performance improvements may be marginal.

**Practical Implementation**

Using Assisted Generation is straightforward, requiring just an additional `--assistant_model` parameter to define the draft model. The draft model generates K tokens, which are evaluated by the target model. The process repeats, ensuring continuous improvement in text generation speed. The acceptance rate of the draft model varies based on input complexity.

Refer to this [tutorial](<insert link to the merged tutorial>) for performing inference on Intel Gaudi 2 Accelerator hardware. This inference method is also compatible with CPUs and multiple Gaudis using DeepSpeed. Originally, this method was referenced from [this blog post](https://huggingface.co/blog/assisted-generation). Habana PyTorch can implement this method for Gaudi accelerators using Optimum Habana as per this [example](https://huggingface.co/blog/assisted-generation). This uses the [asissted_decoding](https://github.com/huggingface/optimum-habana/blob/1e0aa86a58884c46b7b8448b8bbf9654e3d816eb/optimum/habana/transformers/generation/utils.py#L3619) method in the generate class of the text-generation pipeline.


**Conclusion**

The adoption of Assisted Decoding and Speculative Sampling significantly
enhances text generation performance on Intel Gaudi accelerators. These
techniques enable efficient AI inference by reducing latency,
infrastructure costs, and power consumption while maintaining the
quality of generated text. As AI models continue to evolve, optimizing
inference strategies will be key to unlocking their full potential in
real-world applications.

</div>

**References**

\[1\] N. Shazeer, "Fast Transformer Decoding: One Write-Head is All You
Need," Nov. 2019. arXiv:1911.02150.\
\[2\] C. Chen, S. Borgeaud, G. Irving, J.B. Lespiau, L. Sifre, and J.
Jumper, "Accelerating Large Language Model Decoding with Speculative
Sampling," Feb. 2023. arXiv:2302.01318.\
\[3\] J. Gante, "Assisted Generation: a new direction toward low-latency
text generation," May
2023, <https://huggingface.co/blog/assisted-generation>.\
\[4\] <https://huggingface.co/blog/assisted-generation-support-gaudi>
