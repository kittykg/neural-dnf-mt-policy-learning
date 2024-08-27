from dataclasses import dataclass
from enum import IntEnum


# This dataclass is the same as the Atom class 'privately' defined in the
# `neural-ndnf` library post-training and is used for ProbLog interpretation
@dataclass
class Atom:
    id: int
    positive: bool
    type: str  # possible values: "input", "conjunction", "disjunction_head"


class ToyTextSoftExtractionReturnCode(IntEnum):
    # After training
    AFTER_TRAIN_NO_ABNORMAL_STATES = 1  # ideal
    # Pruning codes
    AFTER_PRUNE_NO_ABNORMAL_STATES = 2  # ideal
    FAIL_AT_PRUNE_MISS_ACTION = -1  # failure code
    FAIL_AT_PRUNE_NOT_ME = -2  # failure code
    # Threshold codes
    THRESHOLD_HAS_PERFECT_CANDIDATE = 3  # ideal
    THRESHOLD_IMPERFECT_CANDIDATE = 4  # suboptimal
    # Finish code
    SOFT_EXTRACTION_FINISH = 5


class ToyTextEnvFailureCode(IntEnum):
    # After training failure codes
    FAIL_AT_EVAL_NDNF_TRUNCATED = -1
    FAIL_AT_EVAL_NDNF_EO_TRUNCATED = -2
    FAIL_AT_EVAL_NDNF_MT_TRUNCATED = -3

    # After training EO failure code
    FAIL_AT_EVAL_NDNF_LOSS_PERFORMANCE_AFTER_EO_REMOVED = -4

    # Symbolic interpretation failure codes
    FAIL_AT_EVAL_NDNF_MISS_ACTION = -5
    FAIL_AT_EVAL_NDNF_NOT_ME = -6
    FAIL_AT_EVAL_NDNF_MT_MISS_ACTION = -7
    FAIL_AT_EVAL_NDNF_MT_NOT_ME = -8

    # Pruning failure codes
    FAIL_AT_PRUNE_TRUNCATED = -9
    FAIL_AT_PRUNE_MISS_ACTION = -10
    FAIL_AT_PRUNE_NOT_ME = -11

    # Thresholding failure codes
    FAIL_AT_THRESHOLD_NO_CANDIDATE = -12

    # Rule evaluation failure codes
    FAIL_AT_RULE_EVAL_NOT_ONE_ANSWER_SET = (
        -13
    )  # This theoretically shouldn't happen
    FAIL_AT_RULE_EVAL_MISSING_ACTION = -14
    FAIL_AT_RULE_EVAL_MORE_THAN_ONE_ACTION = -15
    FAIL_AT_RULE_EVAL_TRUNCATED = -16


class SpecialStateCorridorFailureCode(IntEnum):
    # After training failure codes
    FAIL_AT_EVAL_NDNF_TRUNCATED = -1
    FAIL_AT_EVAL_NDNF_EO_TRUNCATED = -2
    FAIL_AT_EVAL_NDNF_MT_TRUNCATED = -3

    # After training EO failure code
    FAIL_AT_EVAL_NDNF_LOSS_PERFORMANCE_AFTER_EO_REMOVED = -4

    # Symbolic interpretation failure codes
    FAIL_AT_EVAL_NDNF_MISS_ACTION = -5
    FAIL_AT_EVAL_NDNF_NOT_ME = -6
    FAIL_AT_EVAL_NDNF_MT_MISS_ACTION = -7
    FAIL_AT_EVAL_NDNF_MT_NOT_ME = -8

    # Pruning failure codes
    FAIL_AT_PRUNE_TRUNCATED = -9
    FAIL_AT_PRUNE_MISS_ACTION = -10
    FAIL_AT_PRUNE_NOT_ME = -11

    # Thresholding failure codes
    FAIL_AT_THRESHOLD_NO_CANDIDATE = -12

    # Rule evaluation failure codes
    FAIL_AT_RULE_EVAL_NOT_ONE_ANSWER_SET = (
        -13
    )  # This theoretically shouldn't happen
    FAIL_AT_RULE_EVAL_MISSING_ACTION = -14
    FAIL_AT_RULE_EVAL_MORE_THAN_ONE_ACTION = -15
    FAIL_AT_RULE_EVAL_TRUNCATED = -16


class DoorCorridorFailureCode(IntEnum):
    # After training failure codes
    FAIL_AT_EVAL_NDNF_TRUNCATED = -1
    FAIL_AT_EVAL_NDNF_LOSS_PERFORMANCE_AFTER_EO_REMOVED = -2
    FAIL_AT_EVAL_NDNF_MT_TRUNCATED = -3

    # Discretisation failure codes
    FAIL_AT_EVAL_NDNF_DIS_TRUNCATED = -4
    FAIL_AT_EVAL_NDNF_MT_DIS_TRUNCATED = -5

    # Symbolic interpretation failure codes
    FAIL_AT_EVAL_NDNF_DIS_MISS_ACTION = -6
    FAIL_AT_EVAL_NDNF_DIS_NOT_ME = -7
    FAIL_AT_EVAL_NDNF_MT_DIS_MISS_ACTION = -8
    FAIL_AT_EVAL_NDNF_MT_DIS_NOT_ME = -9

    # Pruning failure codes
    FAIL_AT_PRUNE_TRUNCATED = -10
    FAIL_AT_PRUNE_MISS_ACTION = -11
    FAIL_AT_PRUNE_NOT_ME = -12

    # Thresholding failure codes
    FAIL_AT_THRESHOLD_NO_CANDIDATE = -13

    # Rule evaluation failure codes
    FAIL_AT_RULE_EVAL_NOT_ONE_ANSWER_SET = (
        -14
    )  # This theoretically shouldn't happen
    FAIL_AT_RULE_EVAL_MISSING_ACTION = -15
    FAIL_AT_RULE_EVAL_MORE_THAN_ONE_ACTION = -16
    FAIL_AT_RULE_EVAL_TRUNCATED = -17
