# ----------------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------
from tensorflow.python.keras.backend import unique_object_name as unique_name
from tensorflow.python.keras.utils.generic_utils import to_snake_case as snake_case


# ----------------------------------------------------------------------------------------------------------------------
# TERM:
# Each neuron network node has an unique name when work in backend. To provide an easy way to debug, plot and configure
# parameters, term produce four types of name of neuron network node into one and provide an access to them.
# ----------------------------------------------------------------------------------------------------------------------
class Term:
    """
    Produce term name of the neuron network node in computation backend.
    ---------
    @author:    Hieu Pham.
    @created:   28th July, 2020.
    """

    def __init__(self, obj=None, prefix=None, name=None, **kwargs):
        """
        Class constructor.
        :param obj:     object need to has a taxonomy.
        :param prefix:  prefix name of taxonomy.
        :param name:    major name of taxonomy.
        :param kwargs:  additional keyword arguments to be passed.
        """
        # Assign prefix name.
        self._prefix = prefix
        # Assign main name.
        self._basename = snake_case(name if name else obj.__class__.__name__)
        # Assign full name.
        self._fullname = '%s.%s' % (self._prefix, self._basename) if self._prefix else self._basename
        self._fullname = unique_name(self._fullname, zero_based=False)
        # Assign proposal name.
        self._proposal = self._fullname.split('.')[-1]
        # Assign suffix name.
        self._suffix = self._proposal.split('_')[-1]
        self._suffix = self._suffix if self._suffix != self._basename else ''

    # Make prefix read-only property.
    @property
    def prefix(self):
        return self._prefix

    # Make basename read-only property.
    @property
    def basename(self):
        return self._basename

    # Make fullname read-only property.
    @property
    def fullname(self):
        return self._fullname

    # Make proposal read-only property.
    @property
    def proposal(self):
        return self._proposal

    # Make suffix read-only property.
    @property
    def suffix(self):
        return self._suffix