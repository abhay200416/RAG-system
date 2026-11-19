from pydantic import BaseModel, Field


class HTML_PARSER(BaseModel):

    """html parser valider """

    url: str = Field(min_length=1, description="html parser")

class PDF_PARSER(BaseModel):
     
    """ pdf parser valider """

    url: str = Field(min_length=1, description="PDF parser")

class TEXT_PARSER(BaseModel):
     
    """text parser valider """

    url: str = Field(min_length=1, description="text parser")

class CHUNKING(BaseModel):

    """chunking validetion"""
    
    docs: str
    chunking_size: int = Field(ge=1 , description="chunking size for docs")
    chunking_overlap: int = Field(ge=1 , description="chunking overlap for docs")


