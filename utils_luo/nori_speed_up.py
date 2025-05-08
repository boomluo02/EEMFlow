import argparse
import warnings
import os
import datetime
import time
import shutil
from collections.abc import Iterable
import pickle

from click import command


# 分屏, 需要先开好一个session, 输入命令list, 自动分屏并执行各个命令
class Tmux_line():
    '''
    get error:  error connecting to /tmp/tmux-11502/default (No such file or directory)
    I meet this error when using this tmux tool in virtual environment (machine by rlaunch or zsh).
    The reason is that the directory /tmp/ is empty in the virtual environment: mkdir /tmp

    Sometimes, there is the created windows may be messy, which is because of the tmux version. Now, this bug is solved.
    '''

    @classmethod
    def new_session(cls, session_name, first_window_name='first'):
        # '''  tmux new-session -s a -n editor -d
        # test:  new_session('a','b')
        # '''
        os.system("tmux new-session -s %s -n %s -d" % (session_name, first_window_name))

    @classmethod
    def new_window(cls, session_name, window_name):
        # '''  tmux neww -a -n tool -t init
        # test:  new_session('a','b')  & new_window('a', 'c')
        # '''
        os.system("tmux neww -a -n %s -t %s" % (window_name, session_name))

    @classmethod
    def switch_window(cls, session_name, window_name):
        # ''' tmux attach -t [session_name]  这个暂时还是别用，会从python弹到tmux对应窗口里面的
        # test:  new_session('a','b')  & new_window('a', 'c') & new_window('a', 'd') & switch_window('a', 'b')
        # '''
        os.system("tmux attach -t %s:%s" % (session_name, window_name))

    @classmethod
    def split_window(cls, session_name, window_name, h_v='h', panel_number=0):
        # ''' tmux split-window -h -t development
        # h表示横着分, v表示竖着分
        # test:  new_session('a','b')  & new_window('a', 'c') & split_window('a', 'b', h_v='h', panel_number=0)
        # '''
        assert h_v in ['h', 'v']
        os.system("tmux split-window -%s -t %s:%s.%s" % (h_v, session_name, window_name, panel_number))

    @classmethod
    def split_window_by_2(cls, session_name, window_name):
        # ''' 拆成2个panel '''
        cls.split_window(session_name, window_name, h_v='v', panel_number=0)  # 上下分两个

    @classmethod
    def split_window_by_4(cls, session_name, window_name):
        # ''' 拆成4个panel '''
        cls.split_window(session_name, window_name, h_v='h', panel_number=0)  # 左右分两个
        cls.split_window(session_name, window_name, h_v='v', panel_number=1)
        cls.split_window(session_name, window_name, h_v='v', panel_number=0)

    @classmethod
    def split_window_by_8(cls, session_name, window_name):
        # ''' 先拆成4个panel '''
        cls.split_window_by_4(session_name, window_name)
        for i in range(4):
            cls.split_window(session_name, window_name, h_v='v', panel_number=3 - i)

    @classmethod
    def split_window_by_16(cls, session_name, window_name):
        # ''' 先拆成8个panel '''
        cls.split_window_by_8(session_name, window_name)
        for i in range(8):
            cls.split_window(session_name, window_name, h_v='h', panel_number=7 - i)

    @classmethod
    def run_command(cls, session_name, window_name, panel_number=0, command_line='ls'):
        com = "tmux send-keys -t %s:%s.%s '%s' C-m" % (session_name, window_name, panel_number, command_line)
        # print(com)
        os.system(com)

    @classmethod
    def _demo(cls):
        # tmux kill-session -t a
        # demo()
        session_name = 'k'
        window_name = 'c'
        cls.new_session(session_name)
        cls.new_window(session_name, window_name)
        cls.split_window_by_16(session_name, window_name)
        for i in range(16):
            time.sleep(0.1)
            cls.run_command(session_name, window_name, i, command_line='ls')

    @classmethod
    def demo_run_commands(cls):
        session_name = 's'
        line_ls = ['ls' for i in range(17)]
        cls.run_task(task_ls=line_ls, task_name='demo', session_name=session_name)

    @classmethod
    def run_command_v2(cls, session_name, window_name, panel_number=0, command_line='ls', **kwargs):
        for i in kwargs.keys():
            command_line += ' --%s %s' % (i, kwargs[i])
        cls.run_command(session_name, window_name, panel_number=panel_number, command_line=command_line)

    @classmethod
    def run_task(cls, task_ls, task_name='demo', session_name='k'):
        # task_ls is a list that contains some string line. Each string line is a command line we want to run.
        N = len(task_ls)
        window_number = 0
        ind = -1

        def create_window(window_number_, panel_number=16):
            window_name = task_name + '_%s' % window_number_
            cls.new_window(session_name, window_name)
            # cls.new_window(session_name, window_name)
            if panel_number == 16:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_16(session_name, window_name)
            elif panel_number == 8:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_8(session_name, window_name)
            elif panel_number == 4:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_4(session_name, window_name)
            elif panel_number == 2:
                print('create a window with %s panels' % panel_number)
                cls.split_window_by_2(session_name, window_name)
            elif panel_number == 1:
                print('create a window with %s panels' % panel_number)
            else:
                pass
            window_number_ += 1
            return window_number_, window_name

        def run_16(data_ls, cnt, window_number_):
            for i in range(len(data_ls) // 16):
                # create window
                window_number_, window_name = create_window(window_number_, panel_number=16)
                print(window_name)
                for j in range(16):
                    cnt += 1
                    if cnt >= N:
                        return cnt, window_number_
                    cls.run_command(session_name=session_name, window_name=window_name, panel_number=j, command_line=data_ls[cnt])
                    print(window_name, data_ls[cnt])

            return cnt, window_number_

        def run_one_window(data_ls, cnt, window_number_, panel_number):
            window_number_, window_name = create_window(window_number_, panel_number=panel_number)
            print(window_name)
            for i in range(panel_number):
                cnt += 1
                if cnt >= N:
                    return cnt, window_number_
                cls.run_command(session_name=session_name, window_name=window_name, panel_number=i, command_line=data_ls[cnt])
                print(window_name, data_ls[cnt])

            return cnt, window_number_

        if N > 16:
            ind, window_number = run_16(task_ls, cnt=ind, window_number_=window_number)
        rest_number = N - ind - 1
        if rest_number > 8:
            ind, window_number = run_one_window(task_ls, cnt=ind, window_number_=window_number, panel_number=16)
        elif rest_number > 4:
            ind, window_number = run_one_window(task_ls, cnt=ind, window_number_=window_number, panel_number=8)
        elif rest_number > 2:
            ind, window_number = run_one_window(task_ls, cnt=ind, window_number_=window_number, panel_number=4)
        elif rest_number > 0:
            ind, window_number = run_one_window(task_ls, cnt=ind, window_number_=window_number, panel_number=2)
        else:
            pass

    @classmethod
    def run_cur_window(cls, data_ls, session_name, window_name, panel_number):
        for i in range(panel_number):
            cls.run_command(session_name=session_name, window_name=window_name, panel_number=i, command_line=data_ls[i])
            print(window_name, data_ls[i])
            # import time
            # time.sleep(30)
        return 

    @classmethod
    def run_cur_command(cls, data_ls, session_name, window_name, panel_ls):
        for i in panel_ls:
            cls.run_command(session_name=session_name, window_name=window_name, panel_number=i, command_line=data_ls[i])
            print(window_name, data_ls[i])

        return 

'''
我想要的是什么样的系统：
1. 执行实验的时候我只用写好main函数, 输入是: 实验文件夹, 实验参数
2. 管理实验的时候，我需要知道，类别分级，结果汇总
'''

if __name__ == '__main__':
    
    session_name = 'speedup'

    speedup_num = 16
    speedup_nori = ''

    command_str = 'conda activate eRaft\n'
    line_ls = [] 
    cls = Tmux_line()
    for s in range(speedup_num):
        command_s = command_str + 'nori speedup ' + speedup_nori +'_' + str(s)+'.nori --on --replica=2'
        line_ls.append(command_s)

    cls.run_task(task_ls=line_ls, task_name='eraft', session_name=session_name)
