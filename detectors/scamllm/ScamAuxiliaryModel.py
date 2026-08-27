from phishnet_feature_engineering.auxiliary_models.AuxiliaryModel import AuxiliaryModel
from pydantic import computed_field
from functools import cached_property
from typing import Any, Dict, List
import torch
from transformers import pipeline


class ScamAuxiliaryModel(AuxiliaryModel):
    """
    Auxiliary model wrapping phishbot/ScamLLM.

    If you use this model in your research, please cite the paper "From Chatbots
    to Phishbots?: Phishing Scam Generation in Commercial Large Language Models"
    (https://www.computer.org/csdl/proceedings-article/sp/2024/313000a221/1WPcYLpYFHy).

    BibTeX:

      @inproceedings{roy2024chatbots,
        title={From Chatbots to Phishbots?: Phishing Scam Generation in Commercial Large Language Models},
        author={Roy, Sayak Saha and Thota, Poojitha and Naragam, Krishna Vamsi and Nilizadeh, Shirin},
        booktitle={2024 IEEE Symposium on Security and Privacy (SP)},
        pages={221--221},
        year={2024},
        organization={IEEE Computer Society}
      }
    """

    model_args: dict = {
        "classifier_id": "phishbot/ScamLLM",
        "task": "text-classification",
        "no_top_k": True,
    }

    @computed_field
    @cached_property
    def classifier(self) -> Any:
        # Check if GPU is available
        device = 0 if torch.cuda.is_available() else -1

        if self.model_args["no_top_k"]:
            classifier = pipeline(
                task=self.model_args["task"],
                model=self.model_args["classifier_id"],
                top_k=None,  # Necessary to retrieve both LABEL_0 and LABEL_1
                device=device,
            )
        else:
            classifier = pipeline(
                task=self.model_args["task"],
                model=self.model_args["classifier_id"],
                device=device,
            )
        return classifier

    def predict(self, message_bodies: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Predict whether each message reads as malicious or safe.

        message_bodies: texts to be evaluated
        returns: one list of {"label", "score"} dicts per input, covering both
                 LABEL_0 and LABEL_1 (because the pipeline runs with top_k=None)

        truncation=True avoids crashes when a generated message is longer than
        the model's 512-token limit.
        """
        return self.classifier(message_bodies, truncation=True, max_length=512)
