# LangChain + SGlang: A First Try

This project is an initial attempt to combine the capabilities of [LangChain](https://github.com/langchain-ai/langchain) and [SGLang](https://github.com/sgl-project/sglang) for enhanced functionality.

## Installation

Follow the steps below to set up the environment and install the necessary dependencies:

1. **Create a Virtual Environment**  
   Set up a virtual environment for your project. 

2. **Check the NVCC Version**  
   Ensure that your system's NVCC version is compatible with the required dependencies.

3. **Install SGlang**  
   Modify the installation link based on your NVCC version. For CUDA 12.4, use the following command:
   ```bash
   pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/ --extra-index-url https://download.pytorch.org/whl/cu124
   ```

4. **Install LangChain**  
   Clone this repository and install LangChain in editable mode:
   ```bash
   git clone <REPO_URL>
   cd PATH/TO/REPO/langchain/libs/langchain
   pip install -e .
   ```

## Supported Features

- **RunnableSequence**: Sequential execution of components.
- **with_structured_output**: Support for structured output formats.

## Unsupported Features

- **RunnableParallel**: Parallel execution is not currently supported.