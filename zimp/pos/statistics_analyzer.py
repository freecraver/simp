from zimp.pos.analyzer import SimpleAggregatedAnalyzer, CountAnalyzer


class TextLengthAnalyzer(SimpleAggregatedAnalyzer):

    def extract_text_metric(self, text: str) -> int:
        return len(text)