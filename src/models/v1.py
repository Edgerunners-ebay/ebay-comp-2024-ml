from typing import Dict, List
from src.data.main import Tags
import dspy
import pydantic


class YearMakeModel(pydantic.BaseModel):
    year: int
    make: str
    model: str


class CompatibleVehicles(pydantic.BaseModel):
    vehicles: List[YearMakeModel]


class CompatibliltyFinderV1(dspy.Signature):
    """
    Find possible vehicles that are compatible to the part based on given information
    """

    description: str = dspy.InputField()
    tags: Tags = dspy.InputField()
    title: str = dspy.InputField()
    output: CompatibleVehicles = dspy.OutputField(
        desc="Year, Make, and Model of the vehicle"
    )


class CompatibliltyFinderV2(dspy.Signature):
    """
    Using the given information about a part, extract all the cars that this part is compatible with, regardless of where this information appears in the data provided.
    Instructions:
    - Search through Items Data, Tags Data, and Description Data to find any mentions of compatible vehicles.
    - Extract the fitment year, make, model for each compatible vehicle.
    - Ignore irrelevant information not related to vehicle compatibility.
    - Use what could be the official names for makes and models
    - Output the results in a structured way
    """

    description: str = dspy.InputField()
    tags: Tags = dspy.InputField()
    title: str = dspy.InputField()
    output: CompatibleVehicles = dspy.OutputField(
        desc="Year, Make, and Model of the vehicle"
    )


class CompatibliltyFinderV3(dspy.Signature):
    """
    Using the given information about a part, extract all the cars that this part is compatible with, regardless of where this information appears in the data provided.
    Instructions:
    - Search through Items Data, Tags Data, and Description Data to find any mentions of compatible vehicles.
    - Extract the fitment year, make, model for each compatible vehicle.
    - Ignore irrelevant information not related to vehicle compatibility.
    - Use what could be the official names for makes and models
    - Output the results in a structured way
    """

    description: str = dspy.InputField(
        desc="Description of the part may contain the years, car models that are compatible with the part"
    )
    tags: Tags = dspy.InputField()
    title: str = dspy.InputField()
    output: CompatibleVehicles = dspy.OutputField(
        desc="Year, Make, and Model of the vehicle, use model as reference and predict make if make is not available"
    )


class ExtractYMM(dspy.Signature):
    """
    Using the given information about a part, extract all the cars that this part is compatible with, regardless of where this information appears in the data provided.
    Instructions:
    - Search through Items Data, Tags Data, and Description Data to find any mentions of compatible vehicles.
    - Extract the fitment year, make, model for each compatible vehicle.
    - Ignore irrelevant information not related to vehicle compatibility.
    - Use what could be the official names for makes and models
    - Give years in the format of YYYY
    - Output the results in a structured way
    """

    description: str = dspy.InputField()
    tags: str = dspy.InputField()
    title: str = dspy.InputField()
    output = dspy.OutputField(
        desc="""List[{'year': int, 'make': str or None, 'model': str or None}]"""
    )


class ExtractYMMV2(dspy.Signature):
    """
    Using the given information about a part, extract all the cars that this part is compatible with, regardless of where this information appears in the data provided.
    Instructions:
    - Search through Items Data, Tags Data, and Description Data to find any mentions of compatible vehicles.
    - Extract the fitment `year`, `make`, `model` for each compatible vehicle.
    - Ignore irrelevant information not related to vehicle compatibility.
    - Use what could be the official names for makes and models
    - Use model name as reference and predict make if make is not available
    - Give years in the format of YYYY
    - Output the results in a structured JSON way

    """

    description: str = dspy.InputField()
    tags: str = dspy.InputField()
    title: str = dspy.InputField()
    output = dspy.OutputField(
        desc="""Eg: {"toyota": {"corolla": [2010, 2011, 2012], "camry": [2010, 2011, 2012]}}"""
    )


class ExtractYMMV3(dspy.Signature):
    """
    Using the given information about a part, extract all the cars that this part is compatible with, regardless of where this information appears in the data provided.
    Instructions:
    - Search through Items Data, Tags Data, and Description Data to find any mentions of compatible vehicles.
    - Extract the fitment `year`, `make`, `model` for each compatible vehicle.
    - Ignore irrelevant information not related to vehicle compatibility.
    - Use what could be the official names for makes and models
    - Use model name as reference and predict make if make is not available
    - Give years in the format of YYYY
    - Include only if absolutely sure about the compatibility
    - Output the results in a structured JSON way
    """

    description: str = dspy.InputField()
    tags: str = dspy.InputField()
    title: str = dspy.InputField()
    output = dspy.OutputField(
        desc="""Eg: {"toyota": {"corolla": [2010, 2011, 2012], "camry": [2010, 2011, 2012]}}"""
    )


class ExtractCarInformation(dspy.Signature):
    """
    Extract the sections that even slightly list vehicle compatibility information from the following data (include "years", car makes, car models), don't generalise makes
    """

    description_data = dspy.InputField()
    output = dspy.OutputField()
