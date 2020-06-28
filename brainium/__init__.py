# ----------------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import absolute_import
from pydesign import Singleton, Locator


# We define a cortex class as a singleton service locator pattern
# which has a responsibility to keep all services of the library.
class Cortex(Locator, metaclass=Singleton):
    pass


# Now we create a cortex instance to work.
cortex = Cortex()
