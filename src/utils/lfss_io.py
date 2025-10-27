import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from lfss.api import Connector
from PIL import Image


@dataclass
class Label:
    @dataclass
    class LabelItem:
        id: str
        low_confidence: float
        description: str
        contours: list[list[float]]
    annotators: list[str]
    overall_description: str
    it: list[LabelItem]

    def compact_json(self) -> dict:
        return {
            "overall_description": self.overall_description,
            "items": [
                {
                    "id": item.id,
                    "description": item.description,
                    "low_confidence": item.low_confidence,
                } for item in self.it
            ]
        }

@dataclass
class Skip:
    reason: str
    skipTime: str

    def compact_json(self) -> dict:
        return {
            "reason": self.reason,
            "skipTime": self.skipTime
        }

@dataclass
class Info:
    file_name: str
    path: str
    source: str

    def compact_json(self) -> dict:
        return {
            "file_name": self.file_name,
            "path": self.path,
            "source": self.source
        }


class ImageReader:
    def __init__(self, image_dir):
        self.c = Connector()
        self.image_dir = image_dir

    def get_raw_image(self, image_id: str) -> Optional[dict]:
        fpath = f"{self.image_dir}/{image_id}.png"
        try:
            image_bytes = self.c.get(fpath)
            if image_bytes is None:
                return None
            return image_bytes
        except Exception as e:
            print(e)
            return None

    def get_encode_image(self, image_id: str) -> Optional[Info]:
        image_bytes = self.get_raw_image(image_id)
        if image_bytes is None:
            return None
        return base64.b64encode(image_bytes).decode("utf-8")
    
    def get_image(self, image_id: str) -> Optional[Info]:
        image_bytes = self.get_raw_image(image_id)
        if image_bytes is None:
            return None
        return Image.open(BytesIO(image_bytes))

class InfoReader:
    def __init__(self, lbl_meta_dir):
        self.c = Connector()
        self.lbl_meta_dir = lbl_meta_dir

    def get_raw_content(self, label_id: str) -> Optional[dict]:
        lbl_fpath = f"{self.lbl_meta_dir}/{label_id}/info.json"
        try:
            content = self.c.get_json(lbl_fpath)
            if content is None:
                return None
            return content
        except Exception as e:
            print(e)
            return None

    def get(self, label_id: str) -> Optional[Info]:
        content = self.get_raw_content(label_id)
        if content is None:
            return None
        
        lbl = Info(
            file_name=content["file_name"],
            path=content["path"],
            source=content["source"]
        )
        return lbl

class LabelReader:
    def __init__(self, lbl_meta_dir):
        self.c = Connector()
        self.lbl_meta_dir = lbl_meta_dir
        
    def get_raw_content(self, label_id: str) -> Optional[dict]:
        lbl_fpath = f"{self.lbl_meta_dir}/{label_id}/label.json"
        try:
            content = self.c.get_json(lbl_fpath)
            if content is None:
                return None
            return content
        except Exception as e:
            return None

    def get(self, label_id: str) -> Optional[Label]:
        content = self.get_raw_content(label_id)
        if content is None:
            return None
        
        lbl = Label(
            annotators=content["annotators"],
            overall_description=content["overallDescription"],
            it=[
                Label.LabelItem(
                    id=item["id"],
                    low_confidence=item["lowConfidence"],
                    description=item["description"],
                    contours=item["contours"]
                ) for item in content["items"]
            ]
        )
        return lbl

class SkipReader:
    def __init__(self, lbl_meta_dir):
        self.c = Connector()
        self.lbl_meta_dir = lbl_meta_dir
        
    def get_raw_content(self, label_id: str) -> Optional[dict]:
        lbl_fpath = f"{self.lbl_meta_dir}/{label_id}/skip.json"
        try:
            content = self.c.get_json(lbl_fpath)
            if content is None:
                return None
            return content
        except Exception as e:
            return None

    def get(self, label_id: str) -> Optional[Skip]:
        content = self.get_raw_content(label_id)
        if content is None:
            return None
        
        lbl = Skip(
            reason=content["reason"],
            skipTime=content["skipTime"]
        )
        return lbl
