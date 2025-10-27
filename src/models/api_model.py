from typing import TypeVar

from openai import OpenAI

from src.models.base_model import BaseModel
from src.utils.common_utils import encode_image

JSON_T = dict | list
JSON_T_VAR = TypeVar("JSON_T_VAR", bound=JSON_T)

class APIModel(BaseModel):
    def __init__(self, model_name: str, temperature: float, base_url: str, api_key: str):
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_from_image_and_text(self, image_path: str, prompt: str, output_type: type[JSON_T_VAR] = dict):
        response = self.client.chat.completions.create(
            extra_body={},
            model=self.model_name,
            messages=[
                {
                "role": "user",
                "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encode_image(image_path)}"
                            }
                        }
                    ]
                }
            ],
            temperature=self.temperature,
        )
        assert (res_text:=response.choices[0].message.content)
        try:
            res = self.t2j(res_text, output_type)
        except Exception as e:
            raise type(e)(f"[{str(e)}] Original Text: {res_text}")
        return res

    def generate_from_text(self, prompt: str, output_type: type[JSON_T_VAR] = dict, temperature: float = None):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ], 
            temperature=temperature if temperature else self.temperature,
        )
        assert (res_text:=response.choices[0].message.content)
        try:
            res = self.t2j(res_text, output_type)
        except Exception as e:
            raise type(e)(f"[{str(e)}] Original Text: {res_text}")
        return res
