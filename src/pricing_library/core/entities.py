from enum import Enum

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class ExerciseStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"
    ASIAN = "asian" 
    BARRIER = "barrier"
    GAP = "gap"