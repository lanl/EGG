""" This module contains classes for constructing models.
"""

class SimData:
    def __mystr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class ObsData:
    def __mystr__(self):
         return str(self.__class__) + ": " + str(self.__dict__)

class Orig:
    def __mystr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class CosmoData:
    def __mystr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class Model:
    def __mystr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class Data:
    def __mystr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class Dist:
    def __mystr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class Priors:
    def __mystr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class Prior:
    def __mystr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class Mcmc:
    def __mystr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class Params:
    def __mystr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

