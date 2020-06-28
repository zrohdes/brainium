# ----------------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2020, the Neuralise authors. All rights reserved.                                                     -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------
from brainium.common import generic


class KwargParser:
    """
    Parses and maps keywords arguments based on schema and keymap
    to ensures the compatibles with Google Tensorlow (Keras).
    ---------
    @author:    Hieu Tr. Pham.
    @created:   28th June, 2020.
    """

    def __init__(self, schema=None, keymap=None, **kwargs):
        """
        Class constructor.
        :param schema: schema of arguments.
        :param keymap: key swapping map of arguments.
        :param kwargs: additional keyword arguments to be passed.
        """
        self._schema = dict(schema) if schema else dict()
        self._keymap = dict(keymap) if keymap else dict()

    def parse(self, nested=False, **kwargs) -> dict:
        """
        Parse arguments from input keyword arguments.
        :param nested:  use nested strategy or not.
        :param kwargs:  input keyword arguments to be parsed.
        :return:        parsed arguments.
        """
        # Initialize things.
        args, temp = self._schema.copy(), {k: kwargs[k] for k in self._schema.keys() & kwargs.keys()}
        # Parse arguments.
        for k in temp:
            checked = generic.isinstances([args[k], temp[k]], dict) and nested
            args.update({k: KwargParser(args[k]).parse(**temp[k]) if checked else temp[k]})
        # Return swapped key arguments.
        return {self._keymap[k] if k in self._keymap else k: args[k] for k in args}