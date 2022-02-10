from zimp.pos.analyzer import SimpleAggregatedAnalyzer


class TextLengthAnalyzer(SimpleAggregatedAnalyzer):

    def extract_text_metric(self, text: str) -> int:
        return len(text)