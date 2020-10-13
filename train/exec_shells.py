import os
if __name__ == '__main__':
    with open('../shells/VSGC_search_full_cora.sh', 'r') as f:
        cmds = f.readlines()
        for cmd in cmds:
            os.system (cmd)