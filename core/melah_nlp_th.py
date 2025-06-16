# melah_nlp_th.py

from pythainlp.tokenize import word_tokenize, sent_tokenize
from pythainlp.tag import pos_tag
from typing import List, Tuple


class MelahNLP_TH:
    def __init__(self):
        self.engine = "deepcut"  # สามารถเปลี่ยนเป็น "newmm" ถ้าต้องการ

    def split_sentences(self, text: str) -> List[str]:
        """แยกประโยคออกจากข้อความ"""
        return sent_tokenize(text)

    def tokenize_words(self, text: str) -> List[str]:
        """แยกคำจากข้อความด้วย engine"""
        return word_tokenize(text, engine=self.engine)

    def tag_pos(self, words: List[str]) -> List[Tuple[str, str]]:
        """ใส่ Part-of-Speech tag ให้แต่ละคำ"""
        return pos_tag(words, corpus="orchid_ud")

    def analyze(self, text: str) -> dict:
        """รวมทุกการวิเคราะห์ไว้ในฟังก์ชันเดียว"""
        sentences = self.split_sentences(text)
        analysis = []

        for sentence in sentences:
            tokens = self.tokenize_words(sentence)
            pos = self.tag_pos(tokens)
            analysis.append({
                "sentence": sentence,
                "tokens": tokens,
                "pos": pos
            })

        return {
            "original": text,
            "sentences": analysis
        }
