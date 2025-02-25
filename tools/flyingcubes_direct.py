import numpy as np
import pyvisa as visa


stl_filepath = '../samples/flyingcubes.stl'
rm = visa.ResourceManager()
print(rm.list_resources())
