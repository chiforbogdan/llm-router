import os
from ray import serve
from ray.serve.llm import LLMConfig, LLMServer, LLMRouter

llm_config1 = LLMConfig(
    model_loading_config=dict(
        model_id="mistral-nemo",
        model_source="casperhansen/mistral-nemo-instruct-2407-awq" ,
        runtime_env={"env_vars": {"USE_VLLM_V1": "0"}}
    ),
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1, 
            initial_replicas=1,
            max_replicas=1,
            max_ongoing_requests=10,
            distributed_executor_backend="ray",
        )
    ),
    engine_kwargs=dict(
        tensor_parallel_size=int(os.environ.get("TENSOR_PARALLELISM", 1)),
        pipeline_parallel_size=int(os.environ.get("PIPELINE_PARALLELISM", 1)),
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", 1024)),
    ),
)

llm_config2 = LLMConfig(
    model_loading_config=dict(
        model_id="llama8b",
        model_source="meta-llama/Llama-3.1-8B-Instruct",
        runtime_env={"env_vars": {"USE_VLLM_V1": "0"}}
    ),
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1,
            initial_replicas=1,
            max_replicas=1,
            max_ongoing_requests=10,
            distributed_executor_backend="ray",
        )
    ),
    engine_kwargs=dict(
        tensor_parallel_size=int(os.environ.get("TENSOR_PARALLELISM", 1)),
        pipeline_parallel_size=int(os.environ.get("PIPELINE_PARALLELISM", 1)),
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", 1024)),
    ),
)

llm_config3 = LLMConfig(
    model_loading_config=dict(
        model_id="code-llama",
        model_source="meta-llama/CodeLlama-7b-Instruct-hf",
        runtime_env={"env_vars": {"USE_VLLM_V1": "0"}}
    ),
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1,
            initial_replicas=1,
            max_replicas=1,
            max_ongoing_requests=10,
            distributed_executor_backend="ray",
        )
    ),
    engine_kwargs=dict(
        tensor_parallel_size=int(os.environ.get("TENSOR_PARALLELISM", 1)),
        pipeline_parallel_size=int(os.environ.get("PIPELINE_PARALLELISM", 1)),
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", 1024)),
    ),
)


deployment1 = LLMServer.as_deployment(llm_config1.get_serve_options(name_prefix="mistral:")).bind(llm_config1)
deployment2 = LLMServer.as_deployment(llm_config2.get_serve_options(name_prefix="llama:")).bind(llm_config2)
deployment3 = LLMServer.as_deployment(llm_config3.get_serve_options(name_prefix="code_llama:")).bind(llm_config3)
llm_app = LLMRouter.as_deployment().bind([deployment1, deployment2, deployment3])