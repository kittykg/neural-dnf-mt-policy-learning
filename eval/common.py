from enum import IntEnum


class FailureCode(IntEnum):
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
