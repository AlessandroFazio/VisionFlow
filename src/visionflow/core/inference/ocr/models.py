from typing import List
import pandas as pd
from visionflow.core.inference.mixins.tesseract import TesseractModelMixin
from visionflow.core.inference.ocr.base import OcrResult, OcrModelBase

import numpy as np
import pytesseract


class PyTesseractModel(OcrModelBase, TesseractModelMixin):
    def extract(self, img: np.ndarray) -> List[OcrResult]:
        df: pd.DataFrame = pytesseract.image_to_data(
            img, lang=self.lang, config=self.config, output_type='data.frame'
        )
        return [
            OcrResult(
                    height=d["height"],
                    width=d["width"],
                    left=d["left"],
                    top=d["top"],
                    page=d["page_num"],
                    paragraph=d["par_num"],
                    line=d["line_num"],
                    confidence=d["conf"],
                    text=d["text"]
                ) for d in df.to_dict('records')
            ]

    @property
    def config(self) -> str:
        return f"--psm {self.psm} -c tessedit_char_whitelist={self.chars_whitelist}"