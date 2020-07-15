#!/usr/bin/env python
# -*- coding: utf-8 -*-


#
# Singleton class -- to store profiler
#

class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):

        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]



class Profiler(object, metaclass=Singleton):

    def __init__(self):
        self._enabled = False


    @property
    def enabled(self):
        return self._enabled


    @enabled.setter
    def enabled(self, val):
        self._enabled = val


    @property
    def profiler(self):

         def identity(f):
            return f
   
        if self.enabled:
            from timemory.profiler import profile
            return profile
        else:
            return identity


    def initialize(self):
        if self.enabled:
            import timemory
            timemory.settings.flat_profile = True
            timemory.settings.timeline_profile = False


    def finalize(self):
        if self.enabled:
            import timemory
            timemory.finalize()

