# import os
# print os.path.dirname(__file__)

import ConfigParser

config = ConfigParser.ConfigParser()
config.read('sentiment_config.ini')
print type(eval(config.get('tokenizing', 'topk')))
print type(eval(config.get('lstm', 'dropout')))
print type(eval(config.get('tokenizing', 'pos_of_tag')))
print eval(config.get('tokenizing', 'pos_of_tag'))