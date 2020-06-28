# ----------------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------


def isinstances(x, types, nested=True) -> bool:
    """
    Check if input object(s) is (an)instances of types or not.
    :param x:       input objects. Can be singular, list and dict.
    :param types:   types that need to be check.
    :param nested:  use nested strategy or not.
    :return:        true or false.
    ---------
    @author:    Hieu Tr. Pham.
    @created:   28th June, 2020.
    """
    # In case of inputs is dict, test all its keys and values.
    if isinstance(x, dict) and nested:
        return all(isinstances(v, types, nested) for k, v in x.items())
    # In case of inputs is list, test all its items.
    elif isinstance(x, list) and nested:
        return all(isinstances(i, types, nested) for i in x)
    # Otherwise, simple test by built-in function.
    return isinstance(x, types)


def content(x, separator=', ', end=' and ', keys=None) -> str:
    """
    Generate string that represent inputs in readable way.
    :param x:           inputs object.
    :param separator:   separator word. Be used if inputs is list or dict.
    :param end:         ending word. Be used if inputs is listable.
    :param keys:        keyword to show if inputs is dict.
    :return:            string that represents inputs.
    ---------
    @author:    Hieu Tr. Pham.
    @created:   28th June, 2020.
    """
    # In case of inputs is dict. Check its keys then treats its as list of contents.
    if isinstance(x, dict):
        arr = keys if isinstances(keys, str) else x.keys()
        return content(['%s=%s' % (arr[k] if k in arr else k, x[k]) for k in arr], separator, end)
    # In case of inputs is list. Make the represent string as below.
    elif isinstance(x, list):
        return '%s%s%s' % (separator.join(x[:-1]), end if len(x) > 1 else '', x[-1])
    # Otherwise, simple cast inputs to string.
    return str(x)