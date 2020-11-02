import os
if __name__ == '__main__':
    with open('../shells/missing_VSGC_Pre_nsl_search_pubmed.sh', 'r') as f:
        cmds = f.readlines()
        for cmd in cmds:
            os.system (cmd)