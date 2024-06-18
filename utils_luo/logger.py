import sys
import os
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        filenameDirs = filename.rsplit("/", 1)

        if len(filenameDirs) > 1:
            if not os.path.exists(filenameDirs[0]):
                os.makedirs(filenameDirs[0])

        self.terminal = stream
        try:
            self.log = open(filename, 'a')
        except:
            print("cant't open the log file {:s}".format(filename))
 
    def write(self, message):
        self.terminal.write(message)
        self.log.truncate() # 清空
        self.log.write(message)
 
    def flush(self):
        pass
 
### example ###
# sys.stdout = Logger(stream=sys.stdout)
# print('print something')
# print("output")