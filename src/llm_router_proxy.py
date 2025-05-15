import ray
from ray import serve
from starlette.requests import Request
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModel, AutoTokenizer
import requests
from enum import Enum
from dataclasses import dataclass

QUERY_CLASSIFIER="nvidia/prompt-task-and-complexity-classifier"

class QueryType(str, Enum):
    OPEN_QA = "Open QA"
    CLOSED_QA = "Closed QA"
    TEXT_GENERATION = "Text Generation"
    CHATBOT = "Chatbot"
    CLASSIFICATION = "Classification"
    SUMMARIZATION = "Summarization"
    CODE_GENERATION = "Code Generation" 
    REWRITE = "Rewrite"
    OTHER = "Other"
    BRAINSTORMING = "Brainstorming"
    EXTRACTION = "Extraction"

class LLMType(str, Enum):
    LLAMA_8B = "llama8b"
    MISTRAL_NEMO = "mistral-nemo"
    CODE_LLAMA = "code-llama"

QUERY_TYPE_LLM = {
    QueryType.OPEN_QA: LLMType.MISTRAL_NEMO,
    QueryType.CLOSED_QA: LLMType.MISTRAL_NEMO,
    QueryType.TEXT_GENERATION: LLMType.MISTRAL_NEMO,
    QueryType.CHATBOT: LLMType.MISTRAL_NEMO,
    QueryType.CLASSIFICATION: LLMType.MISTRAL_NEMO,
    QueryType.SUMMARIZATION: LLMType.MISTRAL_NEMO,
    QueryType.CODE_GENERATION: LLMType.CODE_LLAMA,
    QueryType.REWRITE: LLMType.LLAMA_8B,
    QueryType.OTHER: LLMType.MISTRAL_NEMO,
    QueryType.BRAINSTORMING: LLMType.MISTRAL_NEMO,
    QueryType.EXTRACTION: LLMType.MISTRAL_NEMO,
}

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MulticlassHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
        super(CustomModel, self).__init__()

        self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
        self.target_sizes = target_sizes.values()
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map

        self.heads = [
            MulticlassHead(self.backbone.config.hidden_size, sz)
            for sz in self.target_sizes
        ]

        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)

        self.pool = MeanPooling()

    def compute_results(self, preds, target, decimal=4):
        if target == "task_type":
            task_type = {}

            top2_indices = torch.topk(preds, k=2, dim=1).indices
            softmax_probs = torch.softmax(preds, dim=1)
            top2_probs = softmax_probs.gather(1, top2_indices)
            top2 = top2_indices.detach().cpu().tolist()
            top2_prob = top2_probs.detach().cpu().tolist()

            top2_strings = [
                [self.task_type_map[str(idx)] for idx in sample] for sample in top2
            ]
            top2_prob_rounded = [
                [round(value, 3) for value in sublist] for sublist in top2_prob
            ]

            counter = 0
            for sublist in top2_prob_rounded:
                if sublist[1] < 0.1:
                    top2_strings[counter][1] = "NA"
                counter += 1

            task_type_1 = [sublist[0] for sublist in top2_strings]
            task_type_2 = [sublist[1] for sublist in top2_strings]
            task_type_prob = [sublist[0] for sublist in top2_prob_rounded]

            return (task_type_1, task_type_2, task_type_prob)

        else:
            preds = torch.softmax(preds, dim=1)

            weights = np.array(self.weights_map[target])
            weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
            scores = weighted_sum / self.divisor_map[target]

            scores = [round(value, decimal) for value in scores]
            if target == "number_of_few_shots":
                scores = [x if x >= 0.05 else 0 for x in scores]
            return scores

    def process_logits(self, logits):
        result = {}

        # Round 1: "task_type"
        task_type_logits = logits[0]
        task_type_results = self.compute_results(task_type_logits, target="task_type")
        result["task_type_1"] = task_type_results[0]
        result["task_type_2"] = task_type_results[1]
        result["task_type_prob"] = task_type_results[2]

        # Round 2: "creativity_scope"
        creativity_scope_logits = logits[1]
        target = "creativity_scope"
        result[target] = self.compute_results(creativity_scope_logits, target=target)

        # Round 3: "reasoning"
        reasoning_logits = logits[2]
        target = "reasoning"
        result[target] = self.compute_results(reasoning_logits, target=target)

        # Round 4: "contextual_knowledge"
        contextual_knowledge_logits = logits[3]
        target = "contextual_knowledge"
        result[target] = self.compute_results(
            contextual_knowledge_logits, target=target
        )

        # Round 5: "number_of_few_shots"
        number_of_few_shots_logits = logits[4]
        target = "number_of_few_shots"
        result[target] = self.compute_results(number_of_few_shots_logits, target=target)

        # Round 6: "domain_knowledge"
        domain_knowledge_logits = logits[5]
        target = "domain_knowledge"
        result[target] = self.compute_results(domain_knowledge_logits, target=target)

        # Round 7: "no_label_reason"
        no_label_reason_logits = logits[6]
        target = "no_label_reason"
        result[target] = self.compute_results(no_label_reason_logits, target=target)

        # Round 8: "constraint_ct"
        constraint_ct_logits = logits[7]
        target = "constraint_ct"
        result[target] = self.compute_results(constraint_ct_logits, target=target)

        # Round 9: "prompt_complexity_score"
        result["prompt_complexity_score"] = [
            round(
                0.35 * creativity
                + 0.25 * reasoning
                + 0.15 * constraint
                + 0.15 * domain_knowledge
                + 0.05 * contextual_knowledge
                + 0.05 * few_shots,
                5,
            )
            for creativity, reasoning, constraint, domain_knowledge, contextual_knowledge, few_shots in zip(
                result["creativity_scope"],
                result["reasoning"],
                result["constraint_ct"],
                result["domain_knowledge"],
                result["contextual_knowledge"],
                result["number_of_few_shots"],
            )
        ]

        return result

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        mean_pooled_representation = self.pool(last_hidden_state, attention_mask)

        logits = [
            self.heads[k](mean_pooled_representation)
            for k in range(len(self.target_sizes))
        ]

        return self.process_logits(logits)

class QueryClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(QUERY_CLASSIFIER)
        config = AutoConfig.from_pretrained(QUERY_CLASSIFIER)
        self.model = CustomModel(
            target_sizes=config.target_sizes,
            task_type_map=config.task_type_map,
            weights_map=config.weights_map,
            divisor_map=config.divisor_map,
        ).from_pretrained(QUERY_CLASSIFIER)
        self.model.eval()
        self.model = self.model.to(self.device)
    
    def classify_query(self, query: str) -> str:
        encoded_texts = self.tokenizer(
            query,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        ).to(self.device)
        return self.model(encoded_texts)['task_type_1'][0]


@dataclass
class TokenStats:
    input_tokens: int
    output_tokens: int

@ray.remote
class AccessKeyTokenStats:
    def __init__(self):
        self.data = {}
        print("API usage init")
    
    def set(self, key, data):
        print(f"API usage set {key}")
        self.data[key] = data

    def get(self, key):
        print(f"API usage get {key}")
        return self.data.get(key)

@serve.deployment
class LLMRouterProxy:
    def __init__(self):
        print("Constructor")
        self.query_classifier = QueryClassifier()
        self.tokenizers = {LLMType.LLAMA_8B: AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"),
                           LLMType.CODE_LLAMA: AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-Instruct-hf"),
                           LLMType.MISTRAL_NEMO: AutoTokenizer.from_pretrained("casperhansen/mistral-nemo-instruct-2407-awq")}
        self.stats = ray.get_actor("shared_store")

    async def __call__(self, request: Request):
        body = await request.json()
        
        api_key = body.get("api_key", "")
        if api_key == "":
            return {"message": "Please provide a valid API key"}
        
        query = body.get("query", "")
        if query == "":
            return {"message": "Please provide a valid query"}
        
        query_category = self.query_classifier.classify_query(query)
        print(f"Category is {query_category}")
        
        llm_type = QUERY_TYPE_LLM[QueryType.OTHER]
        if query_category in QUERY_TYPE_LLM:
            llm_type = QUERY_TYPE_LLM[query_category]

        print(f"LLM type {llm_type}")

        resp = requests.post("http://localhost:8000/inference/v1/chat/completions", json={"model": llm_type, "messages": [{"role": "user", "content": query}]})
        print(f"Resp is {resp.json()}")

        model = resp.json()["model"]
        value = resp.json()["choices"][0]["message"]["content"]

        # Get tokens
        input_tokens = self.tokenizers[llm_type].encode(query)
        output_tokens = self.tokenizers[llm_type].encode(value)
        print(f"Input tokens len {len(input_tokens)} output tokens len {len(output_tokens)}")

        api_usage = None
        try:
            api_usage = ray.get(self.stats.get.remote(api_key))
        except:
            print("No API key entry")

        if api_usage is not None:
            token_stats = api_usage.get(llm_type)
            if token_stats is None:
                token_stats = TokenStats(0, 0)
            token_stats.input_tokens += len(input_tokens)
            token_stats.output_tokens += len(output_tokens)
            api_usage[llm_type] = token_stats
        else:
            api_usage = {llm_type: TokenStats(len(input_tokens), len(output_tokens))}
        
        self.stats.set.remote(api_key, api_usage)
        print(f"API usage is: {api_usage}")

        return {model: value}

try:
    AccessKeyTokenStats.options(name="shared_store", lifetime="detached").remote()
except:
    print("Cannot create actor")

llm = LLMRouterProxy.bind()
