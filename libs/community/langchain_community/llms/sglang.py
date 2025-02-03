from typing import Any, Dict, List, Optional, Union
import json
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.utils import pre_init
from pydantic import Field, BaseModel


class Sglang(BaseLLM):
    """Sglang language model."""

    model: str = ""

    """ https://docs.sglang.ai/references/sampling_params.html """
    max_new_tokens: int = 128
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = []
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    n: int = 1

    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None

    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    custom_params: Optional[Dict[str, Any]] = None

    client: Any = None  #: :meta private:

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            import sglang as sgl
        except ImportError:
            raise ImportError(
                "Could not import sglang python package. "
                "Please install it."
            )

        values["client"] = sgl.Engine(
            model_path=values["model"],
        )

        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling sglang."""
        return {
            "n": self.n,
            "max_new_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "json_schema": self.json_schema,
        }

    def _generate(
        self,
        prompts: List[str],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        sampling_params  = {**self._default_params, **kwargs}

        # call the model
        outputs = self.client.generate(prompts, sampling_params)

        generations = []
        for output in outputs:
            text = output["text"]
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "sglang"
    

    def with_structured_output(self, schema: BaseModel) -> "Sglang":
        """Constrain the output to follow a given Pydantic schema."""
        self.json_schema = json.dumps(schema.model_json_schema())
        return self
