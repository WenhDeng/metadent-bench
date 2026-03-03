import base64
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional

from lfss.api import Connector
from PIL import Image


@dataclass
class Label:
    @dataclass
    class LabelItem:
        id: str
        low_confidence: bool
        description: str
        contours: List[List[List[float]]]
    annotators: List[str]
    overall_description: str
    it: List[LabelItem]

    def compact_json(self) -> Dict:
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

    def compact_json(self) -> Dict:
        return {
            "reason": self.reason,
            "skipTime": self.skipTime
        }

@dataclass
class Info:
    file_name: str
    path: str
    source: str

    def compact_json(self) -> Dict:
        return {
            "file_name": self.file_name,
            "path": self.path,
            "source": self.source
        }


class ImageReader:
    def __init__(self, image_dir: str):
        self.c = Connector()
        self.image_dir = image_dir

    def read_as_bytes(
        self, filename: str,
        valid_extensions: set = {'.png', '.jpg', '.jpeg'},
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> bytes:
        assert isinstance(filename, str), \
            f"filename must be a string, got {type(filename).__name__}: {repr(filename)[:50]}"
        assert any(filename.lower().endswith(ext) for ext in valid_extensions), \
            f"File {filename} must be an image file ({', '.join(valid_extensions)})"
        fpath = os.path.join(self.image_dir, filename)
        retry_count = 0

        while retry_count <= max_retries:
            try:
                image_bytes = self.c.get(fpath)
                return image_bytes
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    raise ValueError(f"Max retries exceeded for {fpath}: {e}")
                time.sleep(delay)
                print(f"retrying...")

    def read_as_base64(
        self, filename: str,
        valid_extensions: set = {'.png', '.jpg', '.jpeg'},
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> str:
        image_bytes = self.read_as_bytes(
            filename=filename,
            valid_extensions=valid_extensions,
            max_retries=max_retries,
            delay=delay,
        )
        return base64.b64encode(image_bytes).decode("utf-8")

    def read_as_pil_image(
        self, filename: str,
        valid_extensions: set = {'.png', '.jpg', '.jpeg'},
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> Image.Image:
        image_bytes = self.read_as_bytes(
            filename=filename,
            valid_extensions=valid_extensions,
            max_retries=max_retries,
            delay=delay,
        )
        return Image.open(BytesIO(image_bytes))


class InfoReader:
    def __init__(self, lbl_meta_dir):
        self.c = Connector()
        self.lbl_meta_dir = lbl_meta_dir

    def get_raw_data(
        self, label_id: str | int,
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> Dict:
        """
        Get raw data for a given label id

        :param label_id: The unique identifier of the label
        :type label_id: str | int
        :param max_retries: The maximum number of retries if the request fails
        :type max_retries: int
        :param delay: The delay between retries in seconds
        :type delay: float
        :return: Raw info data for the label
        """
        label_id = str(label_id).zfill(9)
        lbl_fpath = os.path.join(self.lbl_meta_dir, label_id, "info.json")
        retry_count = 0

        while retry_count <= max_retries:
            try:
                content = self.c.get_json(lbl_fpath)
                if content is None:
                    raise FileNotFoundError(f"{lbl_fpath} not found")

                assert isinstance(content, Dict)
                return content
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    raise ValueError(f"Max retries exceeded for {lbl_fpath}: {e}")
                time.sleep(delay)

    def get(
        self, label_id: str | int,
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> Optional[Info]:
        content = self.get_raw_data(
            label_id=label_id,
            max_retries=max_retries,
            delay=delay,
        )

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

    def get_raw_content(
        self, label_id: str | int,
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> Dict | None:
        label_id = str(label_id).zfill(9)
        lbl_fpath = os.path.join(self.lbl_meta_dir, label_id, "label.json")
        retry_count = 0

        while retry_count <= max_retries:
            try:
                content = self.c.get_json(lbl_fpath)
                return content
            except Exception as e:
                if "Not Found" in str(e):
                    return None

                retry_count += 1
                if retry_count > max_retries:
                    raise ValueError(f"Max retries exceeded for {lbl_fpath}: {e}")
                time.sleep(delay)

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

    def get_raw_content(
        self, label_id: str | int,
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> Dict | None:
        label_id = str(label_id).zfill(9)
        lbl_fpath = os.path.join(self.lbl_meta_dir, label_id, "skip.json")
        retry_count = 0

        while retry_count <= max_retries:
            try:
                content = self.c.get_json(lbl_fpath)
                return content
            except Exception as e:
                if "Not Found" in str(e):
                    return None

                retry_count += 1
                if retry_count > max_retries:
                    raise ValueError(f"Max retries exceeded for {lbl_fpath}: {e}")
                time.sleep(delay)

    def get(self, label_id: str) -> Optional[Skip]:
        content = self.get_raw_content(label_id)
        if content is None:
            return None

        lbl = Skip(
            reason=content["reason"],
            skipTime=content["skipTime"]
        )
        return lbl
