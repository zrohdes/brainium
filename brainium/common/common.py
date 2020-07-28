# ----------------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------


class Common:
    """
    Provide common functions for other inherited classes.
    ---------
    @author:    Hieu Pham.
    @created:   28th July, 2020.
    """
    def message(self, message):
        """
        Generate message string with classname itself.
        :param message:     input message.
        :return:            message with classname.
        """
        return '(%s): %s' % (self.__class__.__name__, message)