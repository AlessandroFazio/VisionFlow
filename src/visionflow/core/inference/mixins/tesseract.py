class TesseractModelMixin:
    def __init__(self, psm: int, chars_whitelist: str, lang: str) -> None:
        self.psm = psm
        self.chars_whitelist = chars_whitelist
        self.lang = lang