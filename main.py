import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import logging.config
import yaml
from abc import ABC, abstractmethod

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document


@dataclass
class ExtractorConfig:
    model_name: str
    chunk_size: int
    chunk_overlap: int
    log_level: str
    prompt_templates: Dict[str, str]


class AnalysisType(Enum):
    SUMMARY = "summary"
    KEY_POINTS = "key_points"
    SENTIMENT = "sentiment"


class ContentExtractorError(Exception):
    """base exception for content extractor"""

    pass


class TranscriptError(ContentExtractorError):
    """raised when there's an error fetching transcript"""

    pass


class AnalysisError(ContentExtractorError):
    """raised when there's an error during content analysis"""

    pass


class TranscriptProvider(ABC):
    @abstractmethod
    def get_transcript(self, video_id: str) -> str:
        pass


class YoutubeTranscriptProvider(TranscriptProvider):
    def get_transcript(self, video_id: str) -> str:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([entry["text"] for entry in transcript])
        except Exception as e:
            raise TranscriptError(f"Failed to fetch transcript: {str(e)}")


class ContentAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str, analysis_type: AnalysisType) -> str:
        pass


class LangChainAnalyzer(ContentAnalyzer):

    def __init__(self, config: ExtractorConfig):
        self.config = config
        self.llm = OllamaLLM(model=config.model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )

    def analyze(self, text: str, analysis_type: AnalysisType) -> str:
        try:
            docs = [
                Document(page_content=t) for t in self.text_splitter.split_text(text)
            ]

            if analysis_type == AnalysisType.SUMMARY:
                chain = load_summarize_chain(
                    self.llm, chain_type="map_reduce", verbose=True
                )
            else:
                prompt_template = self.config.prompt_templates[analysis_type.value]
                chain = load_summarize_chain(
                    self.llm,
                    chain_type="stuff",
                    prompt=prompt_template,
                    verbose=True,
                )

            return chain.invoke(docs)
        except Exception as e:
            return AnalysisError(f"Failed to analyze content: {str(e)}")


class ContentExtractor:

    def __init__(self, config_path: Optional[Path] = None):

        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        self.trancript_provider = YoutubeTranscriptProvider()
        self.analyzer = LangChainAnalyzer(self.config)

    def _load_config(self, config_path: Optional[Path]) -> ExtractorConfig:

        default_config = {
            "model_name": "llama2",
            "chunk_size": 2000,
            "chunk_overlap": 200,
            "log_level": "INFO",
            "prompt_templates": {
                "key_points": """
                Extract the main key points from the following text. 
                Format them as a numbered list.
                Text: {text}
                Key Points:
                """,
                "sentiment": """
                Analyze the sentiment of the following text.
                Provide a detailed analysis including:
                - Overall sentiment (positive/negative/neutral)
                - Key emotional themes
                - Supporting evidence from the text
                
                Text: {text}
                Analysis:
                """,
            },
        }

        if config_path and config_path.exists():
            with open(config_path, "w") as file:
                config_data = yaml.safe_load(file)

            default_config.update(config_data)

        return ExtractorConfig(**default_config)

    def _setup_logging(self):

        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "standard": {
                        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    }
                },
                "handlers": {
                    "default": {
                        "level": self.config.log_level,
                        "formatter": "standard",
                        "class": "logging.StreamHandler",
                    },
                },
                "loggers": {
                    __name__: {
                        "handlers": ["default"],
                        "level": self.config.log_level,
                        "propagate": False,
                    },
                },
            }
        )

    @staticmethod
    def extract_video_id(youtube_url: str) -> str:
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtube_url)
        if not video_id_match:
            raise ValueError("invalid youtube url")

        return video_id_match.group(1)

    def process_video(
        self, youtube_url: str, analysis_types: List[AnalysisType]
    ) -> Dict[str, Any]:

        self.logger.info(f"processing video: {youtube_url}")

        try:
            video_id = self.extract_video_id(youtube_url)
            self.logger.debug(f"extracted video id: {video_id}")

            transcript = self.trancript_provider.get_transcript(video_id)
            self.logger.info("successfully retrieved transcript")

            results = {}
            for analysis_type in analysis_types:
                self.logger.info(f"analyzing content for: {analysis_type.value}")
                results[analysis_type.value] = self.analyzer.analyze(
                    transcript, analysis_type
                )
            return results

        except ContentExtractorError as e:
            self.logger.error(f"content extraction error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"unexpected error: {str(e)}")
            raise ContentExtractorError(f"unexpected error: {str(e)}")


if __name__ == "__main__":

    try:
        extractor = ContentExtractor()
        video_url = "https://www.youtube.com/watch?v=WkoytlA3MoQ&ab_channel=AssemblyAI"

        results = extractor.process_video(
            video_url,
            [AnalysisType.SUMMARY, AnalysisType.KEY_POINTS, AnalysisType.SENTIMENT],
        )

        for analysis_type, result in results.items():
            print(f"{analysis_type}: {result}")

    except ContentExtractorError as e:
        logging.error(f"content extraction failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"unexpected error: {str(e)}")
        sys.exit(1)
