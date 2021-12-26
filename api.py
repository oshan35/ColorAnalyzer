from PIL import Image
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PyFileMaker import FMServer
""""
    APIS to be implemented here
"""
fm = FMServer('http://login:password@filemaker.server.com')