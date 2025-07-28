import pandas as pd
import requests
from io import TextIOWrapper, StringIO
from fastapi import HTTPException
import logging

# Create a logger to log errors and info
logger = logging.getLogger(__name__)


class LoadData :
    def __init__(self, file_src=None) :
        # Data source: can be a file, a URL, or None
        self.file_src = file_src

    def load_data(self) -> pd.DataFrame:
        df = None

        # If no source provided, load the default local CSV file
        if self.file_src is None:
            df = pd.read_csv(r"dat/default_data.csv")

        # If the source is an uploaded file
        elif hasattr(self.file_src, 'filename') and hasattr(self.file_src, 'file') :
            try :
                text_stream = TextIOWrapper(self.file_src.file, encoding='utf-8')
                df = pd.read_csv(text_stream)
                self.file_src.file.seek(0)
            except UnicodeDecodeError :
                logger.error(f"File encoding error for {self.file_src.filename}")
                raise HTTPException(status_code=400, detail="UnicodeDecodeError:the file must to be UTF-8 encoded")
            except pd.errors.ParserError as e :
                logger.error(f"CSV Parsing error for {self.file_src.filename}: {e}")
                raise HTTPException(status_code=400,
                                    detail=f"CSV format error: could not parse the file. Details: {e}")
            except Exception as e :
                logger.error(f"Unexpected error during file upload processing: {e}", exc_info=True)
                raise HTTPException(status_code=500,
                                   detail=f"Unexpected server error while processing the URL: {e}")

        # If the source is a URL string starting with http or https
        elif isinstance(self.file_src, str) and self.file_src.startswith(('http://', 'https://')) :
            try :
                response = requests.get(self.file_src)
                response.raise_for_status()
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)
            except requests.RequestException as e :
                raise HTTPException(status_code=400, detail=f"Error downloading file from URL: {e}")
            except Exception as e :
                logger.error(f"Unexpected error during URL processing: {e}", exc_info=True)
                raise HTTPException(status_code=500,
                                    detail=f"An unexpected server error occurred while processing the URL: {e}")

        else :
            # Invalid source type provided
            raise HTTPException(status_code=400, detail=f"Invalid file source type: {type(self.file_src).__name__}")

        # If data was not loaded or is empty, raise an error
        if df is None or df.empty :
            raise HTTPException(status_code=400, detail="Could not load data or the file is empty.")

        df = df.astype(str)
        return df

