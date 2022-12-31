from typing import TypedDict, Union, List, Dict, Tuple

DefaultTypes = Union[int, float, str, List, Dict, Tuple]


class ConfigDict(TypedDict):
    constructor: str
    args: List[DefaultTypes]
    kwargs: List[str, DefaultTypes]
