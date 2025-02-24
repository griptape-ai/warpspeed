---
search:
  boost: 2
---

## Overview

[Image Generation Drivers](../../reference/griptape/drivers/image_generation/index.md) build and execute API calls to image generation models.

Provide a Driver to a [Tool](../tools/index.md) for use by an [Agent](../structures/agents.md):

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/image_generation_drivers_1.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/image_generation_drivers_1.txt"
    ```


## Image Generation Drivers

### Amazon Bedrock

The [Amazon Bedrock Image Generation Driver](../../reference/griptape/drivers/image_generation/amazon_bedrock_image_generation_driver.md) provides multi-model access to image generation models hosted by Amazon Bedrock. This Driver manages API calls to the Bedrock API, while the specific Model Drivers below format the API requests and parse the responses.

#### Stable Diffusion

The [Bedrock Stable Diffusion Model Driver](../../reference/griptape/drivers/image_generation_model/bedrock_stable_diffusion_image_generation_model_driver.md) provides support for Stable Diffusion models hosted by Amazon Bedrock. This Model Driver supports configurations specific to Stable Diffusion, like style presets, clip guidance presets, and sampler.

This Model Driver supports negative prompts. When provided, the image generation request will include negatively-weighted prompts describing features or characteristics to avoid in the resulting generation.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/image_generation_drivers_2.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/image_generation_drivers_2.txt"
    ```


#### Titan

The [Bedrock Titan Image Generator Model Driver](../../reference/griptape/drivers/image_generation_model/bedrock_titan_image_generation_model_driver.md) provides support for Titan Image Generator models hosted by Amazon Bedrock. This Model Driver supports configurations specific to Titan Image Generator, like quality, seed, and cfg_scale.

This Model Driver supports negative prompts. When provided, the image generation request will include negatively-weighted prompts describing features or characteristics to avoid in the resulting generation.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/image_generation_drivers_3.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/image_generation_drivers_3.txt"
    ```


### Azure OpenAI

The [Azure OpenAI Image Generation Driver](../../reference/griptape/drivers/image_generation/azure_openai_image_generation_driver.md) provides access to OpenAI models hosted by Azure. In addition to the configurations provided by the underlying OpenAI Driver, the Azure OpenAI Driver allows configuration of Azure-specific deployment values.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/image_generation_drivers_4.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/image_generation_drivers_4.txt"
    ```


### Leonardo.Ai

The [Leonardo Image Generation Driver](../../reference/griptape/drivers/image_generation/leonardo_image_generation_driver.md) enables image generation using models hosted by [Leonardo.ai](https://leonardo.ai/).

This Driver supports configurations like model selection, image size, specifying a generation seed, and generation steps. For details on supported configuration parameters, see [Leonardo.Ai's image generation documentation](https://docs.leonardo.ai/reference/creategeneration).

This Driver supports negative prompts. When provided, the image generation request will include negatively-weighted prompts describing features or characteristics to avoid in the resulting generation.

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/image_generation_drivers_5.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/image_generation_drivers_5.txt"
    ```


### OpenAI

The [OpenAI Image Generation Driver](../../reference/griptape/drivers/image_generation/openai_image_generation_driver.md) provides access to OpenAI image generation models. Like other OpenAI Drivers, the image generation Driver will implicitly load an API key in the `OPENAI_API_KEY` environment variable if one is not explicitly provided.

This Driver supports image generation configurations like style presets, image quality preference, and image size. For details on supported configuration values, see the [OpenAI documentation](https://platform.openai.com/docs/guides/images/introduction).

=== "Code"
    ```python
    --8<-- "docs/griptape-framework/drivers/src/image_generation_drivers_6.py"
    ```

=== "Logs"
    ```text
    --8<-- "docs/griptape-framework/drivers/logs/image_generation_drivers_6.txt"
    ```


### HuggingFace Pipelines

!!! info

    This driver requires the `drivers-image-generation-huggingface` [extra](../index.md#extras).

The [HuggingFace Pipelines Image Generation Driver](../../reference/griptape/drivers/image_generation/huggingface_pipeline_image_generation_driver.md) enables image generation through locally-hosted models using the HuggingFace [Diffusers](https://huggingface.co/docs/diffusers/en/index) library. This Driver requires a [Pipeline Driver](../../reference/griptape/drivers/image_generation_pipeline/index.md) to prepare the appropriate Pipeline.

This Driver requires a `model` configuration, specifying the model to use for image generation. The value of the `model` configuration must be one of the following:

- A model name from the HuggingFace Model Hub, like `stabilityai/stable-diffusion-3-medium-diffusers`
- A path to the directory containing a model on the filesystem, like `./models/stable-diffusion-3/`
- A path to a file containing a model on the filesystem, like `./models/sd3_medium_incl_clips.safetensors`

The `device` configuration specifies the hardware device used to run inference. Common values include `cuda` (supporting CUDA-enabled GPUs), `cpu` (supported by a device's CPU), and `mps` (supported by Apple silicon GPUs). For more information, see [HuggingFace's documentation](https://huggingface.co/docs/transformers/en/perf_infer_gpu_one) on GPU inference.

#### Stable Diffusion 3 Image Generation Pipeline Driver

!!! info

    The `Stable Diffusion 3 Image Generation Pipeline Driver` requires the `drivers-image-generation-huggingface` extra.

The [Stable Diffusion 3 Image Generation Pipeline Driver](../../reference/griptape/drivers/image_generation_pipeline/stable_diffusion_3_image_generation_pipeline_driver.md) provides a Stable `Diffusion3DiffusionPipeline` for text-to-image generations via the [HuggingFace Pipelines Image Generation Driver's](../../reference/griptape/drivers/image_generation/huggingface_pipeline_image_generation_driver.md) `.try_text_to_image()` method. This Driver accepts a text prompt and configurations including Stable Diffusion 3 model, output image size, generation seed, and inference steps.

Image generation consumes substantial memory. On devices with limited VRAM, it may be necessary to enable the `enable_model_cpu_offload` or `drop_t5_encoder` configurations. For more information, see [HuggingFace's documentation](https://huggingface.co/docs/diffusers/en/optimization/memory) on reduced memory usage.

```python
--8<-- "docs/griptape-framework/drivers/src/image_generation_drivers_7.py"
```

#### Stable Diffusion 3 Img2Img Image Generation Pipeline Driver

!!! info

    The `Stable Diffusion 3 Image Generation Pipeline Driver` requires the `drivers-image-generation-huggingface` extra.

The [Stable Diffusion 3 Img2Img Image Generation Pipeline Driver](../../reference/griptape/drivers/image_generation_pipeline/stable_diffusion_3_img_2_img_image_generation_pipeline_driver.md) provides a `StableDiffusion3Img2ImgPipeline` for image-to-image generations, accepting a text prompt and input image. This Driver accepts a text prompt, an input image, and configurations including Stable Diffusion 3 model, output image size, inference steps, generation seed, and strength of generation over the input image.

```python
--8<-- "docs/griptape-framework/drivers/src/image_generation_drivers_8.py"
```

#### StableDiffusion3ControlNetImageGenerationPipelineDriver

!!! note

    The `Stable Diffusion 3 Image Generation Pipeline Driver` requires the `drivers-image-generation-huggingface` extra.

The [StableDiffusion3ControlNetImageGenerationPipelineDriver](../../reference/griptape/drivers/image_generation_pipeline/stable_diffusion_3_controlnet_image_generation_pipeline_driver.md) provides a `StableDiffusion3ControlNetPipeline` for image-to-image generations, accepting a text prompt and a control image. This Driver accepts a text prompt, a control image, and configurations including Stable Diffusion 3 model, ControlNet model, output image size, generation seed, inference steps, and the degree to which the model adheres to the control image.

```python
--8<-- "docs/griptape-framework/drivers/src/image_generation_drivers_9.py"
```
