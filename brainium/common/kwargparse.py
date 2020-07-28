# ----------------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------
from abc import ABC, abstractmethod
from brainium.common import Common


class KwargParseInterface(ABC, Common):
    """
    Enable keyword arguments parsing of inherited class.
    ---------
    @author:    Hieu Pham.
    @created:   28th July, 2020.
    """
    @abstractmethod
    def parse(self, args=None, **kwargs) -> dict:
        """
        Parse arguments from input keyword arguments.
        :param args:    arguments dict.
        :param kwargs:  keyword arguments to be parsed.
        :return:        arguments dict.
        """
        args = args if isinstance(args, dict) else dict()
        return args


class KwargParseHandle(KwargParseInterface):
    """
    Parse an argument from keyword arguments.
    ---------
    @author:    Hieu Pham.
    @created:   28th July, 2020.
    """
    def __init__(self, name=None, key=None, default=None):
        """
        Class constructor.
        :param name:    name of argument.
        :param key:     key replace of argument.
        :param default: default value of argument.
        """
        # Checking name.
        assert isinstance(name, (str, int)) or name is None, self.message('name must be string or integer.')
        # Checking key.
        assert isinstance(key, (str, int)) or key is None, self.message('key must be string or integer.')
        # Assign variables.
        self.name, self.key, self.default = name, key, default

    def parse(self, args=None, **kwargs):
        """
        Parse arguments from input keyword arguments.
        :param args:    arguments dict.
        :param kwargs:  keyword arguments to be parsed.
        :return:        arguments dict.
        """
        args = super(KwargParseHandle, self).parse(args=args, **kwargs)
        # Do parsing if name is exist.
        if self.name is not None:
            name, default, key = self.name, self.default, self.key if self.key else self.name
            # Check if default value can be parsed ?
            flag = isinstance(default, KwargParseInterface)
            # Update arguments.
            args.update({key: default.parse(**kwargs.get(name, dict())) if flag else kwargs.get(name, default)})
        # Return updated arguments.
        return args


class KwargParse(KwargParseInterface):
    """
    Parse arguments from input keyword arguments.
    ---------
    @author:    Hieu Pham.
    @created:   28th July, 2020.
    """
    def __init__(self):
        """
        Class constructor.
        """
        self.containers = dict()

    def add(self, name=None, key=None, default=None) -> KwargParseInterface:
        """
        Add a kwarg parse handle.
        :param name:    name of handle.
        :param key:     key replacement of handle.
        :param default: default value of handle.
        :return:        kwarg parse itself as facade pattern.
        """
        self.containers[name] = KwargParseHandle(name, key, default)
        return self

    def remove(self, name=None) -> KwargParseInterface:
        """
        Remove a kwarg parse handle.
        :param name:    name of handle to be removed.
        :return:        kwarg parse itself as facade pattern.
        """
        self.containers.pop(name)
        return self

    def parse(self, args=None, **kwargs) -> dict:
        """
        Parse arguments from input keyword arguments.
        :param args:    arguments dict.
        :param kwargs:  keyword arguments to be parsed.
        :return:        arguments dict.
        """
        args = super(KwargParse, self).parse(args=args, **kwargs)
        # Parse arguments.
        for handle in self.containers.values():
            args = handle.parse(args=args, **kwargs)
        # Return parsed arguments.
        return args