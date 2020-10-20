import os
if __name__ == '__main__':
    with open('../shells/GCN_VSGC_result.txt', 'r') as f:
        cmds = f.readlines()
        for cmd in cmds:
            os.system (cmd)